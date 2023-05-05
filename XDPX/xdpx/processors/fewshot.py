import math
import torch
from . import register
from .bert_lm import BertMaskedLMProcessor
from typing import List, Iterator
from xdpx.options import Argument
from xdpx.loaders.fewshot import FewshotLoader
import random


@register('fewshot')
class FewshotProcessor(BertMaskedLMProcessor):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('auxiliary_mlm', default=False),
            Argument('soft_prompt_length', default=0),
            Argument('label_length', default=0),
            Argument('auto_sample_query_cnt', default=0, doc="sample queryset automatically if >0, "
                                                             "useful for mlman,mgimn,etc."
                                                             "if dataset mode is retrieval, ignored "),
        )

    def numerize_episode_mode(self, sample):
        results = {
            'id': sample['id'],
            'text': self.numerize_tokens(sample['text'][:self.args.max_len - self.special_tokens]),
            "label": sample['label'],
            "mode": sample['mode'],
            'domain': sample['domain']
        }
        return results

    def numerize_retrieval_mode(self, sample):
        query_set = sample['query']
        support_set = sample['support']
        query_input_ids = [self.numerize_tokens(tokens[:self.args.max_len - self.special_tokens]) for tokens in
                           query_set]
        support_input_ids = [
            [self.numerize_tokens(tokens[:self.args.max_len - self.special_tokens]) for tokens in items] for
            items in support_set]

        results = {
            'id': sample['id'],
            'query': query_input_ids,
            'support': support_input_ids,
            'query_targets': sample['query_targets'] if 'query_targets' in sample else None,
            "mode": sample['mode'],
            'domain': sample['domain'],
            'support_labels': sample['support_labels'] if 'support_labels' in sample else None
        }
        return results

    def numerize(self, sample: dict):
        if sample['mode'] == FewshotLoader.EPISODE_MODE:
            return self.numerize_episode_mode(sample)
        elif sample['mode'] == FewshotLoader.RETRIEVAL_MODE:
            return self.numerize_retrieval_mode(sample)

    def collate_episode_mode(self, samples):

        labels_list = list(set([s['label'] for s in samples]))
        label_2_index = {label: index for index, label in enumerate(labels_list)}
        support_input_ids = [[] for _ in range(len(labels_list))]

        episode = {'id': [sample['id'] for sample in samples]}

        label_ids = None
        if self.args.soft_prompt_length > 0 and self.args.label_length > 0:
            label_ids = torch.tensor(
                [self.numerize_tokens(list(label)) for label in labels_list],
                # TODO: only support chinese single character
                dtype=torch.long)

        if self.args.auto_sample_query_cnt > 0:
            assert len(samples) > self.args.auto_sample_query_cnt
            random.shuffle(samples)

            query_input_ids = []
            query_targets = []
            for sample in samples[:self.args.auto_sample_query_cnt]:
                if sample['label'] in label_2_index:
                    query_target = label_2_index[sample['label']]
                    query_input_ids.append(sample['text'])
                    query_targets.append(query_target)

            for sample in samples[self.args.auto_sample_query_cnt:]:
                index = label_2_index[sample['label']]
                support_input_ids[index].append(sample['text'])

            support_input_ids, support_input_ids_with_prompt, support_prompt_targets, masked_input_ids, mlm_targets = self.collate_process_support(
                support_input_ids,
                labels_list)
            query_input_ids_tensor, query_targets_tensor, query_input_ids_with_prompt = self.collate_process_query(
                query_input_ids, query_targets)

            episode.update({
                'net_input': {
                    'query_input_ids': query_input_ids_tensor,
                    'support_input_ids': support_input_ids,
                    'query_input_ids_with_prompt': query_input_ids_with_prompt,
                    'support_input_ids_with_prompt': support_input_ids_with_prompt,
                    'masked_input_ids': masked_input_ids,
                    'label_ids': label_ids
                },
                'mlm_targets': mlm_targets,
                'support_prompt_targets': support_prompt_targets,
                'query_targets': query_targets_tensor,
                'ntokens': support_input_ids.numel() + query_input_ids_tensor.numel(),
            })
            return episode

        else:
            for sample in samples:
                index = label_2_index[sample['label']]
                support_input_ids[index].append(sample['text'])

            support_input_ids, support_input_ids_with_prompt, support_prompt_targets, masked_input_ids, mlm_targets = self.collate_process_support(
                support_input_ids,
                labels_list)

            episode.update({
                'net_input': {
                    'query_input_ids': None,
                    'query_input_ids_with_prompt': None,
                    'support_input_ids': support_input_ids,
                    'support_input_ids_with_prompt': support_input_ids_with_prompt,
                    'masked_input_ids': masked_input_ids,
                    'label_ids': label_ids
                },
                'mlm_targets': mlm_targets,
                'support_prompt_targets': support_prompt_targets,
                'ntokens': support_input_ids.numel(),
            })

            return episode

    def collate_retrieval_mode(self, samples):
        assert len(samples) == 1, "when retrievel mode, one episode is one batch, batch_size should be 1"
        support_input_ids = samples[0]['support']
        query_input_ids = samples[0]['query']
        query_targets = samples[0]['query_targets']

        labels_list = samples[0]['support_labels'] if 'support_labels' in samples[0] else None

        episode = {'id': [sample['id'] for sample in samples]}

        label_ids = None
        if self.args.soft_prompt_length > 0 and self.args.label_length > 0 and labels_list:
            label_ids = torch.tensor(
                [self.numerize_tokens(list(label)) for label in labels_list],
                dtype=torch.long)

        if support_input_ids and not query_input_ids:
            # when inference, only compute support embeddings
            support_input_ids, support_input_ids_with_prompt, support_prompt_targets, masked_input_ids, mlm_targets = self.collate_process_support(
                support_input_ids,
                labels_list,
                eval_mode=True)
            episode.update({
                'net_input': {
                    'query_input_ids': None,
                    'support_input_ids': support_input_ids,
                    'query_input_ids_with_prompt': None,
                    'support_input_ids_with_prompt': support_input_ids_with_prompt,
                    'masked_input_ids': masked_input_ids,
                    'label_ids': label_ids
                },
                'mlm_targets': mlm_targets,
                'support_prompt_targets': support_prompt_targets,
                'ntokens': support_input_ids.numel()
            })
            return episode
        elif query_input_ids and not support_input_ids:
            # when inference, only compute query embeddings
            query_input_ids_tensor, query_targets_tensor, query_input_ids_with_prompt = self.collate_process_query(
                query_input_ids, query_targets)
            episode.update({
                'net_input': {
                    'query_input_ids': query_input_ids_tensor,
                    'support_input_ids': None,
                    'masked_input_ids': None,
                    'query_input_ids_with_prompt': query_input_ids_with_prompt,
                    'support_input_ids_with_prompt': None,
                    'label_ids': label_ids
                },
                'query_targets': query_targets_tensor,
                'ntokens': 0
            })
            return episode
        elif support_input_ids and query_input_ids:
            support_input_ids, support_input_ids_with_prompt, support_prompt_targets, masked_input_ids, mlm_targets = self.collate_process_support(
                support_input_ids,
                labels_list)
            query_input_ids_tensor, query_targets_tensor, query_input_ids_with_prompt = self.collate_process_query(
                query_input_ids, query_targets)
            episode.update({
                'net_input': {
                    'query_input_ids': query_input_ids_tensor,
                    'support_input_ids': support_input_ids,
                    'query_input_ids_with_prompt': query_input_ids_with_prompt,
                    'support_input_ids_with_prompt': support_input_ids_with_prompt,
                    'masked_input_ids': masked_input_ids,
                    'label_ids': label_ids
                },
                'mlm_targets': mlm_targets,
                'query_targets': query_targets_tensor,
                'support_prompt_targets': support_prompt_targets,
                'ntokens': support_input_ids.numel() + query_input_ids_tensor.numel(),
            })
            return episode
        else:
            raise RuntimeError("support and query should not be empty at the same time.")

    def collate_process_query(self, query_input_ids, query_targets):
        query_input_ids_tensor = None
        query_input_ids_with_prompt = None
        query_targets_tensor = None
        if query_input_ids:
            if self.args.soft_prompt_length > 0 and self.args.label_length > 0:
                prompt_tokens = ["[unused{}]".format(i + 1) for i in range(self.args.soft_prompt_length)]
                prompt_ids = self.numerize_tokens(prompt_tokens) + [
                    self.args.mask_index] * self.args.label_length

                flattened_input_ids = [
                    [self.args.cls_index] + item + [self.args.sep_index] + prompt_ids + [self.args.sep_index] for item
                    in
                    query_input_ids]
                query_input_ids_with_prompt = torch.tensor(self.pad(flattened_input_ids),
                                                           dtype=torch.long)  # Q * max_seq

            flattened_input_ids = [[self.args.cls_index] + item + [self.args.sep_index] for item in query_input_ids]
            query_input_ids_tensor = torch.tensor(self.pad(flattened_input_ids),
                                                  dtype=torch.long)  # Q * max_seq

        # else: query set and support set are splited automatically and dymatically by model, e.g. BertProtoNet

        if query_targets:
            query_targets_tensor = torch.tensor(query_targets, dtype=torch.long)

        return query_input_ids_tensor, query_targets_tensor, query_input_ids_with_prompt

    def collate_process_support(self, support_input_ids, labels_list=None, eval_mode=False):
        N = len(support_input_ids)
        K = max([len(items) for items in support_input_ids])
        flattened_input_ids = []
        masked_inputs_list = []
        masked_tokens_list = []
        flattened_input_ids_with_prompt = []
        prompt_target_list = []

        if labels_list and self.args.soft_prompt_length > 0 and self.args.label_length > 0:
            assert self.args.auxiliary_mlm, "self.args.auxiliary_mlm should be true to enable prompt"
            for index, items in enumerate(support_input_ids):

                if eval_mode:
                    prompt_tokens = ["[unused{}]".format(i + 1) for i in range(self.args.soft_prompt_length)] + list(
                        labels_list[index])
                    prompt_ids = self.numerize_tokens(prompt_tokens)
                    prompt_targets = [self.args.pad_index] * self.args.soft_prompt_length + self.numerize_tokens(
                        list(labels_list[index]))  # TODO: only support chines

                else:
                    prompt_tokens = ["[unused{}]".format(i + 1) for i in range(self.args.soft_prompt_length)] + [
                        "[MASK]"] * self.args.label_length
                    prompt_ids = self.numerize_tokens(prompt_tokens)
                    prompt_targets = [self.args.pad_index] * self.args.soft_prompt_length + self.numerize_tokens(
                        list(labels_list[index]))  # TODO: only support chinese

                flattened_input_ids.extend(
                    [[self.args.cls_index] + item + [self.args.sep_index] for item in items])
                flattened_input_ids.extend(
                    [[self.args.pad_index] for _ in range(K - len(items))])

                flattened_input_ids_with_prompt.extend(
                    [[self.args.cls_index] + item + [self.args.sep_index] + prompt_ids + [self.args.sep_index] for item
                     in items])
                prompt_target_list.extend(
                    [[self.args.pad_index] + [self.args.pad_index] * (len(item) + 1) + prompt_targets + [
                        self.args.pad_index]
                     for item in
                     items])

            support_input_ids = torch.tensor(self.pad(flattened_input_ids),
                                             dtype=torch.long).reshape(N, K, -1)  # (N, K, len_s)
            support_input_ids_with_prompt = torch.tensor(self.pad(flattened_input_ids_with_prompt), dtype=torch.long)
            support_prompt_targets = torch.tensor(self.pad(prompt_target_list), dtype=torch.long)

        else:
            for index, items in enumerate(support_input_ids):
                flattened_input_ids.extend([[self.args.cls_index] + item + [self.args.sep_index] for item in items])
                flattened_input_ids.extend(
                    [[self.args.pad_index] for _ in range(K - len(items))])
            support_input_ids = torch.tensor(self.pad(flattened_input_ids),
                                             dtype=torch.long).reshape(N, K, -1)  # (N, K, len_s)
            support_input_ids_with_prompt, support_prompt_targets = None, None

        if self.args.auxiliary_mlm:
            for index, items in enumerate(support_input_ids):
                for item in items:
                    masked_inputs, masked_tokens = self.generate_mask(item, None)
                    masked_inputs = [self.args.cls_index] + masked_inputs + [self.args.sep_index]
                    masked_tokens = [self.args.pad_index] + masked_tokens + [self.args.pad_index]
                    masked_inputs_list.append(masked_inputs)
                    masked_tokens_list.append(masked_tokens)
            masked_input_ids = torch.tensor(self.pad(masked_inputs_list), dtype=torch.long)
            mlm_targets = torch.tensor(self.pad(masked_tokens_list), dtype=torch.long)
        else:
            masked_input_ids = None
            mlm_targets = None

        return support_input_ids, support_input_ids_with_prompt, support_prompt_targets, masked_input_ids, mlm_targets

    def collate(self, samples):
        mode = samples[0]['mode']
        episode = None
        if mode == FewshotLoader.EPISODE_MODE:
            episode = self.collate_episode_mode(samples)
        elif mode == FewshotLoader.RETRIEVAL_MODE:
            episode = self.collate_retrieval_mode(samples)
        return episode

    def sanity_check(self, inputs: List[dict]):
        def decode(ids):
            return ' '.join(self.decode(ids)).replace(self.args.pad_word, '_')

        outputs = []
        for i in range(len(inputs)):
            episode = self.collate(inputs[i:i + 1])

            sample = {
                'id': episode['id'],
            }
            net_input = episode['net_input']
            if 'support_input_ids' in net_input:
                sample['support_set'] = []
                for kshot_input_ids in net_input['support_input_ids'].tolist():
                    sample['support_set'].append([decode(input_ids) for input_ids in kshot_input_ids])

            if 'query_input_ids' in net_input and net_input['query_input_ids']:
                sample['query_set'] = [decode(input_ids) for input_ids in
                                       net_input['query_input_ids'].tolist()],

            if 'masked_input_ids' in net_input:
                sample['masked_support_set'] = [decode(input_ids) for input_ids in
                                                net_input['masked_input_ids'].tolist()]
            if 'mlm_targets' in episode:
                sample['mlm_targets'] = [decode(input_ids) for input_ids in
                                         episode['mlm_targets'].tolist()]
            outputs.append(sample)
        return outputs

    @staticmethod
    def target_stream(data: List[dict]) -> Iterator[str]:
        yield 'NotUsedInFewShotTraining'

    def numerize_target(self, sample):
        return {}
