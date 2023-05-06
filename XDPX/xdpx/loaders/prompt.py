from typing import List
from . import register, Loader
from xdpx.options import Argument


@register('prompt_single')
class PromptSingleLoader(Loader):
    _args = None

    def __init__(self, args):
        super(PromptSingleLoader, self).__init__(args)
        self.__class__._args = args

    @property
    def num_sections(self):
        return 2

    @property
    def header(self):
        return ['tokens'] + (['target'] if self.with_targets else [])

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        args = cls._args

        if hasattr(args, 'hard_templates') and args.hard_templates:
            hard_templates = list(args.hard_templates)

            prefix_tokens = []
            for template in hard_templates:
                start_mask_position = template.index("<unk>")

                text1 = template[:start_mask_position]
                text2 = template[start_mask_position + 5:]

                prompt_tokens1 = cls.tokenize(text1)
                prompt_tokens2 = cls.tokenize(text2)

                mask_tokens = ["[MASK]"] * args.label_length
                prefix_tokens.append(prompt_tokens1 + mask_tokens + prompt_tokens2)
        elif hasattr(args, 'soft_prompt_length') and args.soft_prompt_length > 0:
            mask_tokens = ["[MASK]"] * args.label_length
            prompt_tokens = ["[unused{}]".format(i + 1) for i in range(args.soft_prompt_length)]
            prefix_tokens = [prompt_tokens + mask_tokens]
        else:
            raise NotImplementedError("hard prompt or soft prompt, soft_prompt_length>0 or template is not empty")

        return {
            'id': _id,
            'content': cls.tokenize(contents[0]),
            'prefix': prefix_tokens
        }

    def parse_target(self, contents: List[str]) -> dict:
        target = str(contents[-1]).lower()
        assert len(
            target) == self.args.label_length, f"len(target) != self.args.label_length: {target}, {self.args.label_length}"
        return {'target': target, 'target_tokens': list(target)}  # TODO: only support chinese

    def length(self, sample):
        return len(sample['content'])
