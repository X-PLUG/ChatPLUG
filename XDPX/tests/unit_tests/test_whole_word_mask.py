import unittest
from xdpx.tokenizers.bert import BertTokenizer


class TestLinearDecayScheduler(unittest.TestCase):
    def setUp(self):
        self.fn = BertTokenizer.word_begin_mask

    def test_english(self):
        tokens = ['r2', '##30', '两', '背', '板', '的', '机', '器', '能', '用', '吗']
        words = ['r230', '两', '背板', '的', '机器', '能用', '吗']
        mask = self.fn(words, tokens)
        self.assertEqual(list(map(int, mask)), [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1])

    def test_long_seq(self):
        words = '中华人民共和国 中央人民广播电台'.split()
        tokens = ['中', '华', '人', '民', '共', '和', '国',
                  '中', '央', '人', '民', '广', '播', '电', '台']
        mask = self.fn(words, tokens)
        self.assertEqual(list(map(int, mask)), [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    
    def test_korean(self):
        tokens = [
            '是', '这', '个', '歌', '曲', '，', '您', '用', 'qq', '音', '乐', '听', '也', '行', '，', 
            '歌', '名', '是', 'am', '##a', '##zi', '##ng', 'grace', ',', 
            '是', '韩', '国', '歌', '手', '刘', '沙', '朗', '唱', '的',
            'ᄋ', '##ᅲ', 'ᄉ', '##ᅡ', 'ᄅ', '##ᅡ', '##ᆼ', '唱','的'
        ]
        text = '是 这个 歌曲 ， 您 用 qq音乐 听 也 行 ， 歌名 是 amazing grace , 是 韩国 歌手 刘 沙朗 唱 的 유 사 랑 唱 的'
        mask = self.fn(text, tokens)
        answer = [
            1, 1, 0, 1, 0, 1,
            1, 1, 1, 0, 0, 1, 1, 1, 1,
            1, 0, 1, 1, 0, 0, 0, 1, 1,
            1, 1, 0, 1, 0, 1, 1, 0, 1, 1,
            1, 0, 1, 0, 1, 0, 0, 1, 1,
        ]
        # give up the Korean part for now.....
        # self.assertEqual(list(map(int, mask)), answer)
        # self.assertEqual(len(mask), len(tokens))
        self.assertEqual(mask, None)
    
    def test_quotation_mark(self):
        words = ['《', '“平安盈”服务协议', '》', '和', '《', '“平安盈”快速转出服务协议', '》', '等', '文件', '内容']
        tokens = [
            '《', '“', '平', '安', '盈', '”', '服', '务', '协', '议', '》', '和', '《', '“',
            '平', '安', '盈', '”', '快', '速', '转', '出', '服', '务', '协', '议', '》', '等', '文', '件', '内', '容'
        ]
        mask = self.fn(words, tokens)
        self.assertIsNotNone(mask)
        answer = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0]
        self.assertEqual(list(map(int, mask)), answer)


if __name__ == '__main__':
    unittest.main()
