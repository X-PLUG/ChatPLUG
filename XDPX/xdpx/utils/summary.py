import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from transformers.file_utils import ModelOutput


def summary_string(model, dummy_inputs):
    def register_hook(module):
        def hook(module, input, output):

            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()

            if isinstance(input, (list, tuple)):
                summary[m_key]["input_shape"] = [
                    list(o.size()) if torch.is_tensor(o) else [] for o in input
                ]
            elif torch.is_tensor(input):
                summary[m_key]["input_shape"] = list(input.size())
            else:
                summary[m_key]["input_shape"] = []

            if isinstance(output, ModelOutput):
                output = output.to_tuple()

            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    list(o.size()) if torch.is_tensor(o) else [] for o in output
                ]
            elif torch.is_tensor(output):
                summary[m_key]["output_shape"] = list(output.size())
            else:
                summary[m_key]["output_shape"] = []

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))

    x = dummy_inputs

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str = ""
    summary_str += "----------------------------------------------------------------" + "\n"

    inputs_shape_str = "inputs size = "
    for input in dummy_inputs:
        if input is None:
            inputs_shape_str += 'None '
        else:
            inputs_shape_str += str(input.shape) + ' '

    summary_str += inputs_shape_str + '\n'

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        output_shape = summary[layer]["output_shape"]
        if isinstance(output_shape, (list, tuple)):
            for o in output_shape:
                total_output += np.prod(o)
        else:
            total_output += np.prod(output_shape)

        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + " when " + inputs_shape_str + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)
