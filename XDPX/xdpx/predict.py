import os
import sys
import torch
from tqdm import tqdm
from typing import Dict
from xdpx.options import Options, Argument, Arg
from xdpx.utils import io, move_to_cuda
from xdpx.utils.profiling import measure_runtime
from xdpx.evaluate import Evaluator, load_input_data
from xdpx.bootstrap import bootstrap


def cli_main(argv=sys.argv):
    options = Options()
    Evaluator.register(options)
    options.register(
        Argument('predict_file_map', required=True, type=Dict[str, str], 
                 doc='files need to be predicted and their corresponding saved names',
                 validate=lambda value: all(io.exists(key) or key.startswith('odps://') for key in value.keys())),
        Argument('distill', default=False, doc='whether to distill logits only', children={
            lambda val: not val: [
                Argument('header', default=True),
            ]
        }),
        Argument('binary', default=False, doc='store predicted results in binary files.'),
        Argument('max_predict', type=int),
        Argument('concat_origin', default=True),
    )
    options.add_global_constraint(lambda args: not (args.skip_bad_lines and args.concat_origin))
    options.add_global_constraint(lambda args: not (args.concat_origin and args.binary))
    options.add_global_constraint(lambda args: (
        not (args.concat_origin and any(file.endswith('.pt') for file in args.predict_file_map.keys())),
        'cannot use concat_origin with binary files'
    ))
    bootstrap(options, main, __file__, argv)


def main(cli_args: Arg):
    args, task, model, loss, loader = Evaluator.build_evaluation(cli_args)
    task.processor.target_map._unk_index = 0

    for infile, outfile in cli_args.predict_file_map.items():
        data, origin_data = load_input_data(infile, loader, task.processor)
        if origin_data is None and cli_args.concat_origin:
            raise RuntimeError('The binary input format is not compatible with "concat_origin"')
        data = data[:cli_args.max_predict]
        data = task.build_dataset(data, is_train=False)
        preds = []
        interrupted = False
        runtime = []
        with tqdm(data, desc='predicting') as progress:
            try:
                for sample in progress:
                    if cli_args.cuda:
                        sample = move_to_cuda(sample)
                    with measure_runtime(lambda t: runtime.append(t)):
                        if cli_args.distill:
                            pred = task.distill_step(sample, model, loss)
                        else:
                            pred = task.inference_step(sample, model, loss)
                    preds.append(pred)
            except KeyboardInterrupt:
                interrupted = True
        print(f'| prediction time per batch: {sum(runtime) / len(runtime):.3f}ms')

        pred_lines = []
        for pred in tqdm(preds, desc='formatting'):
            lines = zip(*pred)
            for line in lines:
                if not cli_args.binary:
                    line = '\t'.join(map(str, line))
                pred_lines.append(line)
        print(f'| generate {len(pred_lines)} predictions in total.')
        if cli_args.binary:
            with io.open(outfile, 'wb') as f:
                torch.save(pred_lines, f)
            continue
        io.makedirs(os.path.dirname(outfile), exist_ok=True)
        with io.open(outfile, 'w') as f:
            if hasattr(cli_args, 'header') and cli_args.header:
                header = task.inference_header
                if cli_args.concat_origin:
                    header = loader.header + header
                f.write('\t'.join(header) + '\n')
            if not cli_args.concat_origin:
                for pred_line in pred_lines:
                    f.write(f'{pred_line}\n')
            else:
                if interrupted:
                    print(f'prediction interrupted. Only save the first {len(pred_lines)} predictions...')
                else:
                    assert len(origin_data[:cli_args.max_predict]) == len(pred_lines), \
                        f'{len(origin_data[:cli_args.max_predict])}, {len(pred_lines)}'
                shift_target = cli_args.distill and (loader.with_targets and
                                                     len(origin_data[0]) == loader.num_sections)
                for orig_line, pred_line in zip(origin_data, pred_lines):
                    if not shift_target:
                        f.write('\t'.join(orig_line) + f'\t{pred_line}\n')
                    else:
                        f.write('\t'.join(orig_line[:-1]) + f'\t{pred_line}\t{orig_line[-1]}\n')


if __name__ == "__main__":
    cli_main()
