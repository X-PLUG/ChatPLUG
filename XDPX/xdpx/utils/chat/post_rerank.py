import torch
import os
import traceback
from typing import List, Tuple
from xdpx.tasks import tasks
from xdpx.loaders import loaders
from xdpx.options import Options, Arg
from xdpx.utils import io, move_to_cuda, parse_model_path
import torch.nn.functional as F


def convert_to_local_cfg_vocab(args):
    config = {
        'hfl/chinese-roberta-wwm-ext': 'tests/sample_data/hfl_chinese-roberta-wwm-ext.json'
    }
    args.p_auto_config = config.get(args.p_auto_config, args.p_auto_config)
    args.q_auto_config = config.get(args.q_auto_config, args.q_auto_config)
    return args


class PostReranker:
    def __init__(self, save_dir, checkpoint=None):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.cuda else torch.device('cpu')

        with io.open(os.path.join(save_dir, 'args.py')) as f:
            args = Options.parse_tree(eval(f.read()))
        try:
            with io.open(os.path.join(args.data_dir, 'args.py')) as f:
                args = Arg().update(Options.parse_tree(eval(f.read()))).update(args)
        except IOError:
            pass
        args = convert_to_local_cfg_vocab(args)
        args.__cmd__ = 'serve'
        args.save_dir = save_dir
        args.temperature = 1.0
        # build the task
        task = tasks[args.task](args)
        model = task.build_model(args)
        loss = task.build_loss(args)
        model_path = checkpoint if checkpoint else parse_model_path('<best>', args)
        model.load(model_path)

        if self.cuda:
            model = model.cuda()

        self.processor = task.processor
        self.loader = loaders[args.loader](args)
        self.task = task
        self.model = model
        self.loss = loss
        self.args = args

    def rerank(self, query, responses: List[str]) -> List[Tuple[str, float]]:
        '''

        Args:
            query:
            responses:

        Returns:
            list of tuple(response, score)
        '''
        if not responses:
            return []
        batch = []
        for response in responses:
            batch.append([query, response])
        try:
            batch = [self.processor.encode(self.loader, sample) for sample in batch]
            batch = self.processor.collate(batch)

            if self.cuda:
                batch = move_to_cuda(batch)

            self.model.eval()
            with torch.no_grad():
                z1, z2 = self.model(**batch['net_input'])
                cos_sim = F.cosine_similarity(z1, z2, dim=-1, eps=1e-4).tolist()  # batch_size
                results = []
                for i, response in enumerate(responses):
                    results.append((response, cos_sim[i]))
                results = sorted(results, key=lambda x: x[1], reverse=True)
                return results
        except Exception as e:
            traceback.print_exc()
