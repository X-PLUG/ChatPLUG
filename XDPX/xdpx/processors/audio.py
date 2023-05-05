import torch
import os
# import torchaudio
from . import register, Processor
from .bert import BertSingleProcessor
from xdpx.options import Argument
from xdpx.utils import io
from transformers import Wav2Vec2FeatureExtractor


class Wav2vecProcessor(Processor):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('audio_dir', type=str),
            Argument('wav2vec_processor', type=str),
            Argument('max_audio_length', type=int, default=80000) # 5 seconds
        )

    def __init__(self, args):
        super().__init__(args)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            pretrained_model_name_or_path=self.args.wav2vec_processor)


@register('bert_single_wav2vec')
class BertSingleWav2vecProcessor(Wav2vecProcessor, BertSingleProcessor):
    def numerize(self, inputs: dict):
        text = self.numerize_tokens(inputs['tokens'])

        audio_file = os.path.join(self.args.audio_dir, inputs['audio'])
        with io.open(audio_file, 'rb') as f:
            audio_input_values, _ = torchaudio.load(f)
            audio_input_values = audio_input_values.numpy()[0][:self.args.max_audio_length]  # numpy.ndarray

        results = {
            'id': inputs['id'],
            'input_ids': [self.args.cls_index, *text[:self.args.max_len - 2], self.args.sep_index],
            'audio_file': inputs['audio'],
            'audio_input_values': audio_input_values
        }
        return results

    def collate(self, samples):

        input_ids = torch.tensor(self.pad([sample['input_ids'] for sample in samples]), dtype=torch.long)
        speeches = [sample['audio_input_values'] for sample in samples]

        audio_features = self.feature_extractor(speeches, sampling_rate=16000, padding=True, return_tensors='pt')

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
                'audio_input_values': audio_features['input_values'],
                'audio_attention_mask': audio_features['attention_mask']
            },
            'ntokens': input_ids.numel(),
        }
        try:
            target = torch.tensor([sample['target'] for sample in samples], dtype=torch.long)
            batch.update({'target': target})
        except KeyError:
            ...
        return batch
