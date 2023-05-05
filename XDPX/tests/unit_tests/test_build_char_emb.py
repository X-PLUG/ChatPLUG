import torch
import unittest
from xdpx.models.bert_mix import BertMix


class TestBertMix(unittest.TestCase):
    def test_build_char_emb(self):
        cases = [
            {
                'tokens': [
                    [1, 2, 3, 4],
                    [5, 6, 0, 0],
                    [7, 8, 9, 0],
                ],
                'mask': [
                    [1, 0, 1, 0],
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                ],
                'answer': [
                    [[1, 2], [3, 4]],
                    [[5, 6], [0, 0]],
                    [[7, 0], [8, 9]],
                ],
            },
            {
                'tokens': [
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [9, 10, 11, 12, 13, 14, 0, 0],
                ],
                'mask': [
                    [1, 1, 1, 0, 1, 0, 0, 0],
                    [1, 0, 0, 1, 1, 0, 0, 0],
                ],
                'answer': [
                    [[1, 0, 0, 0], [2, 0, 0, 0], [3, 4, 0, 0], [5, 6, 7, 8]],
                    [[9, 10, 11, 0], [12, 0, 0, 0], [13, 14, 0, 0], [0, 0, 0, 0]],
                ],
            },
        ]
        for case in cases:
            tokens = torch.tensor(case['tokens'])
            hidden_size = 2
            tokens_mask = tokens.ne(0).unsqueeze(2)
            tokens = torch.stack([tokens] * hidden_size, 2)
            word_begins_mask = torch.tensor(case['mask'])
            output, _ = BertMix.build_char_emb(
                tokens, tokens_mask, word_begins_mask)
            ref = torch.tensor(case['answer'])
            ref = torch.stack([ref] * hidden_size, 3)
            self.assertTrue(output.eq(ref).all().item())


if __name__ == '__main__':
    unittest.main()
