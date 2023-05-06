import logging
import json
import sys
from tqdm import tqdm
from xdpx.utils import io
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import time

'''
usage: x-script read_open_dialog_finetune_data <src_dir> <target_dir>
'''
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.WARNING)


def process_batch(batch, tokenizer, model):
    inputs = tokenizer(batch, return_tensors="pt", padding=True).to(torch.device('cuda'))
    with torch.no_grad():
        outputs = model(**inputs)
    results = []
    for index, (batch_i, start_score, end_score, input_id) in enumerate(
            zip(batch, outputs.start_logits, outputs.end_logits, inputs.input_ids)):
        max_startscore = torch.argmax(start_score)
        max_endscore = torch.argmax(end_score)
        ans_tokens = inputs.input_ids[index][max_startscore: max_endscore + 1]
        answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens,
                                                        skip_special_tokens=True)
        answer_span = tokenizer.convert_tokens_to_string(answer_tokens).replace(' ', '')

        item = {'question': batch_i[0], 'passage': batch_i[1],
                'answer_span': answer_span}
        results.append(item)
    return results


def cli_main(argv):
    assert len(argv) == 3, 'x-script read_open_dialog_finetune_data <src_dir> <target_dir>'
    src_dir, target_dir = argv[1], argv[2]
    files = [f for f in io.listdir(src_dir) if 'train' in f or 'dev' in f]
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-pert-base-mrc')
    model = BertForQuestionAnswering.from_pretrained('hfl/chinese-pert-base-mrc').cuda()
    for file in files:
        print(file)
        with io.open(src_dir + '/' + file) as f:
            data = [json.loads(l) for l in f.readlines()]
        with io.open(target_dir + '/' + file, 'w') as nf:
            for d in tqdm(data):
                question = d['context'].strip().strip('</s>').split('</s>')[-1]
                passages = d.get('passages')
                batch = []
                if passages:
                    passages = [p for p in passages.split(';;;') if p.strip()]
                    new_passages = []
                    type_list = []
                    for passage in passages:
                        if 'history' in passage:
                            new_passages.append(passage.replace('\t', '</s>'))
                        elif ':' in passage:
                            index = passage.index(':')
                            batch.append((question, passage[index + 1:]))
                            type_list.append(passage[:index])
                    results = process_batch(batch, tokenizer, model)
                    for item, type in zip(results, type_list):
                        span = item["answer_span"]
                        if span:
                            passage = item['passage'].replace(span, f'<em>{span}</em>')
                        else:
                            passage = item['passage']
                        new_passages.append(f'{type}: {passage}')
                    new_passages = ';;;'.join(new_passages)
                    nd = {'context': d['context'], 'response': d['response'], 'passages': new_passages}
                    nf.write(json.dumps(nd, ensure_ascii=False) + '\n')
                else:
                    nf.write(json.dumps(d, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    cli_main(sys.argv[1:])
