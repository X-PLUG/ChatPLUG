import unittest
from xdpx.options import Arg
from xdpx.tokenizers.char_zh import ChineseCharacterTokenizer


class TestChineseCharacterTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ChineseCharacterTokenizer(Arg(rm_punc=False, lower=True))

    def test_tokenize(self):
        text = '物流 显示 ， 只有 一个 号 ， 400811696093	帮 我 看 一下 我 的 快递 到 哪里 了'
        tokens = ['物', '流', '显', '示', '，', '只', '有', '一', '个', '号', '，', 'NUM12', '帮', '我', '看',
                  '一', '下', '我', '的', '快', '递', '到', '哪', '里', '了']
        self.assertEqual(tokens, self.tokenizer.encode(text))

        text = '《 从你的全世界路过—特别故事集 》 那个 。'
        tokens = ['《', '从', '你', '的', '全', '世', '界', '路', '过', '—', '特',
                  '别', '故', '事', '集', '》', '那', '个', '。']
        self.assertEqual(tokens, self.tokenizer.encode(text))

        text = '6-18个月的这个有多长	裙摆有多长 ？？	'
        tokens = ['6', '-', '18', '个', '月', '的', '这', '个', '有', '多', '长', '裙', '摆', '有', '多', '长', '？']
        self.assertEqual(tokens, self.tokenizer.encode(text))

        text = 'emmmm,我想问一下这件衣服的颈圈是得自己另外买吗'
        tokens = ['emmmm', ',', '我', '想', '问', '一', '下', '这', '件', '衣', '服', '的', '颈', '圈',
                  '是', '得', '自', '己', '另', '外', '买', '吗']
        self.assertEqual(tokens, self.tokenizer.encode(text))


if __name__ == '__main__':
    unittest.main()
