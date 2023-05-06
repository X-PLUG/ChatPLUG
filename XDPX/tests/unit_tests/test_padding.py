import torch
import unittest
from xdpx.options import Arg


class TestPadding(unittest.TestCase):
    def test_RE2(self):
        args = Arg()
        args.data_dir = '.'
        args.__cmd__ = 'x-train'
        args.embedding_dim = 10
        args.hidden_size = 10
        args.blocks = 2
        args.dropout = 0.0
        args.fix_embeddings = False
        args.encoder_layers = 2
        args.kernel_sizes = [1, 3]
        args.vocab_size = 20
        args.num_classes = 2
        args.alignment = 'linear'
        args.prediction = 'full'

        torch.manual_seed(321)
        if torch.cuda.is_available:
            torch.cuda.manual_seed(321)
        from xdpx.models.re2 import RE2
        model = RE2(args)
        text1 = torch.randint(1, args.vocab_size, (4, 7))
        text2 = torch.randint(1, args.vocab_size, (4, 5))
        mask1 = torch.ne(text1, 0)
        mask2 = torch.ne(text2, 0)
        result1 = model(text1, text2, mask1, mask2)

        text1 = torch.cat([text1, torch.zeros(4, 8).to(text1)], dim=1)
        text2 = torch.cat([text2, torch.zeros(4, 5).to(text2)], dim=1)
        mask1 = torch.ne(text1, 0)
        mask2 = torch.ne(text2, 0)
        result2 = model(text1, text2, mask1, mask2)

        self.assertTrue(torch.allclose(result1, result2))


if __name__ == '__main__':
    unittest.main()
