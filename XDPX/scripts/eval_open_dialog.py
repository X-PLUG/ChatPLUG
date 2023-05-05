import logging
import json
import sys
from tqdm import tqdm
from xdpx.options import Options, Argument, Arg
from xdpx.bootstrap import bootstrap
from xdpx.utils.chat_serve import QARankModel, EnsembleModel
from xdpx.utils import io

'''
usage: x-script eval_open_dialog <test_file_path>
'''
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def score_qa(query, response, model):
    search_results = [{'snippet': response}]
    results = model.rerank(query, search_results)
    score = results[0]['qa_score']
    return score

def cli_main(argv):
    assert len(argv) == 2 or len(argv) == 3, 'x-script eval_open_dialog <test_file_path> <metric_model_dir>'
    test_file_path = argv[1]
    metric_model_dir = argv[2] if len(
        argv) > 2 else 'TODO'

    if not metric_model_dir == 'ensemble':
        metric_model = QARankModel(metric_model_dir)
    else:
        metric_model = EnsembleModel()

    try:
        data = json.load(io.open(test_file_path))
        data = data['results']
    except:
        data = [json.loads(l) for l in io.open(test_file_path).readlines()[1:]]
    for d in tqdm(data):
        context = d['history']
        query = d['utterance']
        response = d['generated_response']
        d['score'] = score_qa(query, response, metric_model)
    scores = [d['score'] for d in data]
    final_score = sum(scores) / len(scores)
    print(f'samples cnt: {len(scores)} \t final_score: {final_score}')
    type_scores = {}
    for d in data:
        type = d['type']
        if type not in type_scores:
            type_scores[type] = []
        type_scores[type].append(d['score'])
    for k, v in type_scores.items():
        score = sum(v) / len(v)
        print(f'type: {k} \t score: {score}')


if __name__ == '__main__':
    cli_main(sys.argv[1:])
