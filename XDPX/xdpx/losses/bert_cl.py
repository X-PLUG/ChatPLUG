import torch
import torch.nn as nn
from . import register
from .bert import BertLoss
from xdpx.options import Argument
from xdpx.tasks import Task, register as register_task
import json
from itertools import chain
from sklearn.metrics import accuracy_score
import numpy as np
from xdpx.loaders import loaders
from xdpx.utils import io, move_to_cuda
import traceback
from xdpx.utils.distributed_utils import is_master, should_barrier
import os
import numpy as np
import logging
from scipy.stats import spearmanr
import math
from typing import List
import torch.distributed as dist
from xdpx.utils import io, cache_file, get_train_subsets
from xdpx.utils.io_utils import tqdm, CallbackIOWrapper
import random


@register('bert_cl')
class BertCLLoss(BertLoss):
    """
    Contrastive Learning
    """

    @staticmethod
    def register(options):
        BertLoss.register(options)
        options.register(
            Argument('supervised', default=False, doc='whether the approach is supervised or not'),
            Argument('negative_proto_size', default=256, doc='number of sampled negative prototypes'),
            Argument('lambda_proto_loss', default=0.01),
        )

    def forward(self, model, sample, protocl=False, cluster_result=None, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert reduce is True
        orig_input_ids = sample['orig_input_ids']
        batch_size = int(orig_input_ids.size(0))

        input_ids = torch.cat([orig_input_ids] * 2, dim=0)  # [bsz * 2, sent_len]
        orig_attention_mask = orig_input_ids.ne(self.padding_idx)
        attention_mask = torch.cat([orig_attention_mask] * 2, dim=0)

        last_hidden_states, pooled_output, embeddings = model(input_ids, masked_tokens=attention_mask)

        pooler_output = torch.split(pooled_output, batch_size, dim=0)

        z1, z2 = pooler_output[0], pooler_output[1]

        # Gather all embeddings if using distributed training
        if dist.is_initialized() and model.training:
            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        logging_output = {}

        # infonce loss
        cos_sim = model.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        max_p, indices = torch.max(cos_sim, dim=1)
        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)

        loss_proto = 0
        duplicate_pos = 0

        if protocl and cluster_result:
            z1 = torch.nn.functional.normalize(z1)  # batch_size x hidden_size
            hard_negatives = []
            for i, (im2cluster, prototypes, density) in enumerate(
                    zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])):
                # prototypes: cluster_size x hidden_size, density: cluster_size
                index = sample['id']
                proto_id = im2cluster[index]
                hard_negatives.append(prototypes[proto_id])

            z3 = torch.cat(hard_negatives, dim=0)
            z1_z3_cos = model.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

            batch_size = cos_sim.size(0)
            weights = []
            for i in range(batch_size):
                row = [0.0] * batch_size
                for j in range(len(self.args.num_cluster)):
                    row += [0.0] * i + [3] + [0.0] * (batch_size - i - 1)
                weights.append(row)
            weights = torch.tensor(weights)
            cos_sim = cos_sim + weights

            loss = loss_fct(cos_sim, labels)

        logging_output.update({
            'loss': loss.item(),
            'duplicate_pos': duplicate_pos,
            'loss_proto': loss_proto,
            'target': labels.tolist(),
            'pred': indices.tolist()
        })

        # loss and other stats are already averaged, so 1 is used as the sample_size
        return loss, batch_size, logging_output

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        target = list(chain.from_iterable(log['target'] for log in logging_outputs))[:sample_size]
        pred = list(chain.from_iterable(log['pred'] for log in logging_outputs))[:sample_size]

        agg_output = dict({
            'sample_size': sample_size
        })

        agg_output['loss'] = sum(log['loss'] for log in logging_outputs) / sample_size
        agg_output['loss_proto'] = sum(log['loss_proto'] for log in logging_outputs) / sample_size
        agg_output['duplicate_pos'] = sum(log['duplicate_pos'] for log in logging_outputs)

        accuracy = accuracy_score(target, pred)
        agg_output['acc'] = accuracy

        if 'spearman' in logging_outputs[0]:
            agg_output['spearman'] = max(log['spearman'] for log in logging_outputs)

        if 'ntokens' in logging_outputs[0]:
            ntokens = sum(log['ntokens'] for log in logging_outputs)
            agg_output['ntokens'] = ntokens
        return agg_output

    def inference(self, model, sample):
        orig_input_ids = sample['orig_input_ids']
        orig_attention_mask = orig_input_ids.ne(self.padding_idx)
        a, b, embedding = model(orig_input_ids, masked_tokens=orig_attention_mask)
        return embedding


@register_task('bert_cl')
class BertCLTask(Task):
    FOR_CLUSTER = 'for_cluster'

    @staticmethod
    def register(options):
        Task.register(options)
        options.register(
            Argument('path_to_stsb', default=None, type=str,
                     doc='stsbenchmark data root dir; if not None , use senteval tool and stsbenchmark dataset in valid step'),
            Argument('cluster_warmup', default=0, type=int, doc='warmup steps before first clustering'),
            Argument('cluster_interval', default=10000, type=int),
            Argument('num_cluster', default=[], type=List[int],
                     doc='when empty, use infonce same as simcse, otherwise use protonce'),
            Argument('cluster_niter', default=50, type=int),
            Argument('cluster_nredo', default=3, type=int),
            Argument('max_points_per_centroid', default=1000, type=int),
            Argument('min_points_per_centroid', default=10, type=int),
            Argument('data_size_for_cluster', default=10000, type=int, doc='subset of train dataset for clustering'),

        )
        options.add_global_constraint(
            lambda args: not args.num_cluster or args.data_size_for_cluster > 0
        )

        from xdpx.loaders import Loader
        Loader.register(options)

    def __init__(self, args):
        super().__init__(args)
        self.loader = loaders['corpus'](args)
        if args.path_to_stsb:
            self.se = STSBenchmarkEval(self.args.path_to_stsb)
        self.cluster_result = None

    def eval_stsb_spearman(self, model, loss):
        def batcher(batch):
            sentences = [[' '.join(s)] for s in batch]
            batch = [self.processor.encode(self.loader, sample) for sample in sentences]
            batch = self.processor.collate(batch)
            if torch.cuda.is_available():
                batch = move_to_cuda(batch)

            with torch.no_grad():
                embeddings = loss.inference(model, batch)
                results = np.array(embeddings.cpu())
                return results

        print("| Start STSBenchmarkEval...")
        spearman = self.se.run(batcher)
        print("| End STSBenchmarkEval, result = {}".format(spearman))
        return spearman

    def load_dataset(self, splits, is_train, reload=False):
        self.load_dataset_for_clustering()
        if self.args.path_to_stsb and not is_train:
            self.data_loaders[self.VALID] = [[1]]  # A trick, Trainer.validate will exploit it in a loop
        else:
            return super().load_dataset(splits, is_train, reload)

    def load_dataset_for_clustering(self):
        if self.FOR_CLUSTER in self.data_loaders:
            return self.data_loaders[self.FOR_CLUSTER]

        splits = get_train_subsets(self.args)
        if isinstance(splits, str):
            splits = [splits]
        data = []
        for split in splits:
            if split not in self.datasets:
                path = os.path.join(self.args.data_dir, f'{split}.pt')
                if self.args.cache_train_file:
                    path = cache_file(path)
                if os.path.exists(path):
                    md5sum = io.md5(path)
                    print(f'| Loading dataset {split} ({md5sum})')
                data_fsize = io.size(path)
                with tqdm(total=data_fsize, unit='B', unit_scale=True, unit_divisor=1024,
                          leave=data_fsize > 100 * 1024 ** 2,  # 100M
                          desc=f'reading {split}') as t, io.open(path, 'rb') as obj:
                    obj = CallbackIOWrapper(t.update, obj, "read")
                    data_i = torch.load(obj)
                self.datasets[split] = data_i
            data.extend(self.datasets[split])
            # if len(data) >= self.args.data_size_for_cluster:
            #     break
        data = data[:self.args.data_size_for_cluster]
        data = self.build_dataset(data, False)
        self.data_loaders[self.FOR_CLUSTER] = data
        return data

    def run_clustering(self, model, loss):
        cluster_data_loader = self.load_dataset_for_clustering()
        model.eval()
        loss.eval()
        features = torch.zeros(len(cluster_data_loader.dataset), self.args.hidden_size).cuda()
        print('len(cluster_data_loader.dataset):{}'.format(len(cluster_data_loader.dataset)))

        for i, batch in enumerate(tqdm(cluster_data_loader)):
            with torch.no_grad():
                if torch.cuda.is_available():
                    batch = move_to_cuda(batch)
                embeddings = loss.inference(model, batch)
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                index = batch['id']
                features[index] = embeddings

        if should_barrier():
            dist.barrier()
            dist.all_reduce(features, op=dist.ReduceOp.SUM)
        features = features.cpu()

        cluster_result = {'im2cluster': [], 'centroids': [], 'density': []}
        for num_cluster in self.args.num_cluster:
            cluster_result['im2cluster'].append(torch.zeros(features.size(0), dtype=torch.long).cuda())
            cluster_result['centroids'].append(torch.zeros(int(num_cluster), self.args.hidden_size).cuda())
            cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())
        if is_master(self.args):
            features = features.numpy()
            cluster_result = self.kmeans(features)

        if should_barrier():
            dist.barrier()
            # broadcast clustering result
            for k, data_list in cluster_result.items():
                for data_tensor in data_list:
                    dist.broadcast(data_tensor, 0, async_op=False)
        self.cluster_result = cluster_result

    def kmeans(self, x):
        import faiss
        print('performing kmeans clustering')
        results = {'im2cluster': [], 'centroids': [], 'density': []}

        for seed, num_cluster in enumerate(self.args.num_cluster):
            # intialize faiss clustering parameters
            d = x.shape[1]
            k = int(num_cluster)
            clus = faiss.Clustering(d, k)
            clus.verbose = True
            clus.niter = self.args.cluster_niter
            clus.nredo = self.args.cluster_nredo
            clus.seed = seed
            clus.max_points_per_centroid = self.args.max_points_per_centroid
            clus.min_points_per_centroid = self.args.min_points_per_centroid

            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = 0
            index = faiss.GpuIndexFlatL2(res, d, cfg)

            clus.train(x, index)

            D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
            im2cluster = [int(n[0]) for n in I]

            # get cluster centroids
            centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

            # sample-to-centroid distances for each cluster
            Dcluster = [[] for c in range(k)]
            for im, i in enumerate(im2cluster):
                Dcluster[i].append(D[im][0])

            # concentration estimation (phi)
            density = np.zeros(k)
            for i, dist in enumerate(Dcluster):
                if len(dist) > 1:
                    d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                    density[i] = d

            # if cluster only has one point, use the max to estimate its concentration
            dmax = density.max()
            for i, dist in enumerate(Dcluster):
                if len(dist) <= 1:
                    density[i] = dmax

            density = density.clip(np.percentile(density, 10),
                                   np.percentile(density, 90))  # clamp extreme values for stability
            print('num_cluster:{} \t density: {}'.format(num_cluster, density))

            density = self.args.temperature * density / density.mean()  # scale the mean to temperature
            # convert to cuda Tensors for broadcast
            centroids = torch.Tensor(centroids).cuda()
            centroids = nn.functional.normalize(centroids, p=2, dim=1)
            im2cluster = torch.LongTensor(im2cluster).cuda()
            density = torch.Tensor(density).cuda()

            results['centroids'].append(centroids)
            results['density'].append(density)
            results['im2cluster'].append(im2cluster)

        return results

    def train_step(self, sample, model, loss, optimizer, num_updates=0):
        protocl = False
        cluster_result = self.cluster_result
        if self.args.num_cluster and num_updates > self.args.cluster_warmup:
            protocl = True

        if self.args.num_cluster and num_updates % self.args.cluster_interval == 0 \
                and num_updates + self.args.cluster_interval > self.args.cluster_warmup:
            self.run_clustering(model, loss)

        model.train()
        loss.train()
        loss, sample_size, logging_output = loss(model, sample, protocl, cluster_result)
        optimizer.backward(loss)
        logging_output['ntokens'] = sample['ntokens']

        if self.args.inspect_gradient and num_updates % self.args.eval_interval == 0:
            from xdpx.visualize import plot_grad_flow
            plot_grad_flow(
                model.named_parameters(),
                os.path.join(self.args.save_dir, 'plots', 'gradients', f'{num_updates}.' + self.args.figext)
            )

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, loss):
        model.eval()
        loss.eval()
        if self.args.path_to_stsb:
            if is_master(self.args):
                spearman = self.eval_stsb_spearman(model, loss)
            else:
                spearman = 0

            logging_output = {
                'loss': 0.1,
                'loss_proto': 0.1,
                'duplicate_pos': 0,
                'spearman': spearman,
                'target': [1],  # A trick, will be used in aggregate_logging_outputs
                'pred': [1]  # A trick, will be used in aggregate_logging_outputs

            }
            return 0, 1, logging_output
        else:
            with torch.no_grad():
                loss, sample_size, logging_output = loss(model, sample)
            return loss, sample_size, logging_output

    def inference_step(self, sample, model, loss):
        # original text
        model.eval()
        orig_text = [
            ' '.join(self.processor.decode(sample_i.tolist())) for
            sample_i in
            sample['orig_input_ids']]
        with torch.no_grad():
            embeddings = loss.inference(model, sample)

        return orig_text, embeddings

    @property
    def inference_header(self):
        return 'ori_input embedding'.split()

    @staticmethod
    def newline_concat(texts):
        # use double quotes to escape a single quote in csv
        return '"' + ('\n'.join(texts)).replace('"', '""') + '"'


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


class STSBenchmarkEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.dev_data = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        self.batch_size = 64

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

    def run(self, batcher):
        sys_scores = []
        input1, input2, gs_scores = self.dev_data
        for ii in tqdm(range(0, len(gs_scores), self.batch_size)):
            batch1 = input1[ii:ii + self.batch_size]
            batch2 = input2[ii:ii + self.batch_size]

            # we assume get_batch already throws out the faulty ones
            if len(batch1) == len(batch2) and len(batch1) > 0:
                enc1 = batcher(batch1)
                enc2 = batcher(batch2)

                for kk in range(enc2.shape[0]):
                    sys_score = self.similarity(enc1[kk], enc2[kk])
                    sys_scores.append(sys_score)
        results = spearmanr(sys_scores, gs_scores)
        return results[0]
