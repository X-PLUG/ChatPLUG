import torch
from functools import partial
from xdpx.utils import register
from xdpx.options import Argument
import random
from . import register


@register('episode')
class EpisodeSampler(torch.utils.data.BatchSampler):
    @classmethod
    def register(cls, options):
        options.register(
            Argument('n_way', default=10),
            Argument('k_shot', default=5),
            Argument('train_episodes', default=500),
            Argument('dev_episodes', default=50),
            Argument('include_domains', default=None),
        )

    def __init__(self, args, dataset, *kargs, **kwargs):
        self.args = args
        self.dataset = dataset
        self.epoch = 0

        domain_label_sampleid = {}
        for sample_index, sample in enumerate(dataset):
            label = sample['label']
            domain = sample['domain']
            if args.include_domains is not None and domain not in args.include_domains:
                continue
            label_tokens = domain_label_sampleid.get(domain, {})
            domain_label_sampleid[domain] = label_tokens
            sample_list = label_tokens.get(label, [])
            label_tokens[label] = sample_list
            sample_list.append(sample_index)
        self.domain_label_tokens = domain_label_sampleid
        self.domains = list(domain_label_sampleid.keys())
        domain_size = len(self.domains)
        print("| domains: {} {}".format(self.domains[:10],
                                        '... total count:{}'.format(domain_size) if domain_size > 10 else ''))
        all_labels_size = 0
        for domain, label_tokens in domain_label_sampleid.items():
            all_labels_size += len(label_tokens)
        print("| all labels size:{}".format(all_labels_size))

    def __iter__(self):
        if self.dataset.is_train:
            num_episodes = self.args.train_episodes
            rand = random
        else:
            num_episodes = self.args.dev_episodes
            rand = random.Random(0)  # fixed seed

        for _ in range(num_episodes):
            domain_id = rand.randint(a=0, b=len(self.domains) - 1)
            domain = self.domains[domain_id]
            all_labels = self.domain_label_tokens[domain].keys()
            n_way = min(self.args.n_way, len(all_labels))
            labels = rand.sample(population=all_labels, k=n_way)
            batch = []
            for label in labels:
                k = min(self.args.k_shot, len(self.domain_label_tokens[domain][label]))
                batch.extend(rand.sample(population=self.domain_label_tokens[domain][label], k=k))
            yield batch

    def __len__(self):
        if self.dataset.is_train:
            return self.args.train_episodes
        else:
            return self.args.dev_episodes

    def set_epoch(self, epoch):
        self.epoch = epoch
