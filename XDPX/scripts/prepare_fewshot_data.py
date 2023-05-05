import os
import sys
import traceback
import torch
from prettytable import PrettyTable
from xdpx.tasks import tasks
from xdpx.loaders import loaders
from xdpx.utils import io, move_to_cuda, parse_model_path
from xdpx.bootstrap import bootstrap
from xdpx.options import Options, Argument, Arg
import logging
from typing import List
import json

import numpy as np
import random

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('data_dir', required=True, type=str),
        Argument('save_dir', required=True, type=str),
        Argument('k_shot', default=3),
        Argument('n_way', default=5),
        Argument('train_episodes', default=100),
        Argument('dev_episodes', default=50),
        Argument('q_shot', default=0),
        Argument('tasks', required=True, type=str),

    )
    bootstrap(options, main, __file__, argv)


def main(args: Arg):
    tasks = args.tasks.split(',')
    for split in ['train', 'dev', 'test']:
        episodes = []

        for task in tasks:
            ori_path = os.path.join(args.data_dir, task, split + '.json')
            if io.exists(ori_path):
                samples = json.loads(io.open(ori_path).read())
                dic = {}
                for sample in samples:
                    label = sample['label']
                    text = sample['text']
                    if label not in dic:
                        dic[label] = []
                    dic[label].append(text)
                labels = list(dic.keys())

                num_episodes = args.train_episodes if split == 'train' else args.dev_episodes
                for _ in range(num_episodes):
                    random.shuffle(labels)
                    support_set = []
                    support_set2 = []
                    query_set = []
                    for l in labels[:args.n_way]:
                        items = dic[l]
                        random.shuffle(items)
                        support_set.append(items[:args.k_shot])
                        if args.q_shot > 0:
                            query_set.append(items[args.k_shot:args.k_shot + args.q_shot])
                            support_set2.append([item + ' ' + l for item in items[:args.k_shot]])

                    episodes.append({'support_set': support_set, 'query_set': query_set, 'support_set2': support_set2})

        random.shuffle(episodes)
        args.save_dir = args.save_dir.strip('/')

        save_path = os.path.join(args.save_dir + '_False', split + '.json')
        with io.open(save_path, 'w') as f:
            json.dump([{'support_set': e['support_set'], 'query_set':e['query_set']} for e in episodes], f, indent=3)

        if args.q_shot > 0:
            save_path = os.path.join(args.save_dir + '_True', split + '.json')
            with io.open(save_path, 'w') as f:
                json.dump([{'support_set': e['support_set2'], 'query_set': e['query_set']} for e in episodes], f, indent=3)


if __name__ == '__main__':
    cli_main(sys.argv)
