from typing import List
from xdpx.options import Argument
from . import Loader, register


@register('session_corpus')
class SessionCorpusLoader(Loader):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('min_session_size', default=8),
            Argument('max_consecutive', default=2),
        )
        options.assume_defined('max_messages', type=int, by=cls.__name__)

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        return {
            'id': _id,
            'msg_no': int(contents[0]),
            'sender': int(contents[1]),
            'content': cls._tokenizer.encode(contents[2]),
        }

    @property
    def header(self):
        return ['sender', 'content']

    def length(self, sample):
        return max(map(len, sample['content']))

    def load_data(self, path: str, ordered='auto') -> List[dict]:
        return super().load_data(path, True)

    def merge(self, samples: List[dict]) -> List[dict]:
        sessions = []
        prev_msg_no = -1
        buffer = []

        def submit_buffer():
            if buffer:
                # concat short messages from the same sender
                senders = []
                contents = []
                prev_sender = -1
                for _, sender, content in buffer:
                    if sender != prev_sender or not contents or len(contents[-1]) + len(content) > self.args.max_len:
                        prev_sender = sender
                        senders.append(sender)
                        contents.append(content)
                    else:
                        contents[-1] += content
                # filter short messages that are usually not informative
                assert len(senders) == len(contents)
                senders, raw_senders = [], senders
                contents, raw_contents = [], contents
                for sender, content in zip(raw_senders, raw_contents):
                    if len(content) > self.args.min_len:
                        senders.append(sender)
                        contents.append(content)
                # filter consecutive messages that are mostly redundant
                senders, raw_senders = [], senders
                contents, raw_contents = [], contents
                combo = 0
                prev_sender = -1
                for sender, content in zip(raw_senders, raw_contents):
                    if sender == prev_sender:
                        combo += 1
                    else:
                        combo = 0
                        prev_sender = sender
                    if combo >= self.args.max_consecutive:
                        continue
                    senders.append(sender)
                    contents.append(content)
                if len(contents) >= self.args.min_session_size and len(set(senders)) > 1:
                    sessions.append(dict(
                        id=buffer[0][0],
                        sender=senders,
                        content=contents,
                    ))
                buffer.clear()

        for sample in samples:
            msg_no = sample['msg_no']
            if msg_no < prev_msg_no:
                submit_buffer()
            prev_msg_no = msg_no
            buffer.append((sample['id'], sample['sender'], sample['content']))
        submit_buffer()
        return sessions
