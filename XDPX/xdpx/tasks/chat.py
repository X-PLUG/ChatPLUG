import torch
from . import register, Task
from xdpx.options import Argument
from xdpx.loaders import loaders

# @register('vincent_chat')
# class ChatTask(Task):
#     def __init__(self, args):
#         super(ChatTask, self).__init__(args)
#         self.tokenizer = loaders[self.args.loader](self.args).tokenizer.tokenizer
#
#     def inference_step(self, sample, model, loss):
#         model.eval()
#         loss.eval()
#         with torch.no_grad():
#             responses, scores = loss.inference(model, sample, self.tokenizer)
#         return responses, scores
#
#     @property
#     def inference_header(self):
#         return ['generated_response', 'score']
