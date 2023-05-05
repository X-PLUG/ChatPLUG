import os
import sys
import traceback
import torch
from xdpx.options import Arg, Options
from xdpx.tasks import tasks
from xdpx.loaders import loaders
from xdpx.loaders.parsers import parsers
from xdpx.utils import io, move_to_cuda, parse_model_path
import json
from xdpx.utils.chat_serve import QARankModel,EnsembleModel
from rouge import Rouge


def cli_main(argv):
    assert len(argv) == 2
    file_path = argv[1]
    model = QARankModel('oss://xdp-expriment/gaoxing.gx/chat/training/rerank/v0.9.1.rerank/1024_0.0001')
    lines = open(file_path).readlines()
    data = [json.loads(l) for l in lines]
    rouge = Rouge()
    n_cands = 4
    for d in data:
        query = d['context'].strip('</s>').split('</s>')[-1]
        response = d['response'].strip().strip('</s>')
        if not response:
            continue
        passages = [p for p in d['passages'].split(';;;') if p]
        gen_candidates = [{'snippet': t} for t in set(d['gen_candidates'][:n_cands])]
        use_rouge_for_rerank = False
        for passage in passages:
            rouge_score = rouge.get_scores(' '.join(response), ' '.join(passage))[0]
            rouge_p = rouge_score['rouge-l']['p']
            if rouge_p > 0.9:
                use_rouge_for_rerank = True
                break

        if use_rouge_for_rerank:
            for candidate in gen_candidates:
                rouge_score = rouge.get_scores(' '.join(candidate['snippet']), ' '.join(response))[0]
                rouge_f = rouge_score['rouge-l']['f']
                candidate['score'] = rouge_f
            rerank_results = sorted(gen_candidates, key=lambda x: x['score'], reverse=True)
            d['rank_type'] = 'rouge_score'
        else:
            rerank_results = model.rerank(query, gen_candidates)
            d['rank_type'] = 'metric_model'

        d['candidates'] = ';;;'.join([t['snippet'] for t in rerank_results])
        print(json.dumps(d, ensure_ascii=False))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python xdpx/chat_rerank_candidates.py file')
        exit()
    cli_main(sys.argv)
