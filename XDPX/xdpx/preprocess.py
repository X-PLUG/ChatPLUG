import json
import os
import sys
import random
import torch
import hjson
from typing import List, Dict, Union, Optional
from fnmatch import fnmatch
from datetime import datetime
from collections import OrderedDict
import xdpx
from xdpx.options import Options, Argument
from xdpx.utils import io, log_to_file, download_from_url, validate_url
from xdpx.processors import processors
from xdpx.loaders import loaders
from xdpx.dictionary import Dictionary
from xdpx.bootstrap import bootstrap


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('data_source', required=True, doc='path for original data', validate=lambda value: io.isdir(value)),
        Argument('data_files', default='*.txt', type=Union[Dict[str, str], List[str], str],
                 doc='the filename pattern for data files to be processed.'),
        Argument('data_dir', required=True),
        Argument('vocab_file', doc='predefined vocab file', children={
            lambda value: value is None: [
                Argument('threshold', default=-1, doc='threshold when creating vocab', children={
                    lambda value: value > 0: [
                        Argument('ignore_in_emb', default=True,
                                 doc='if a word appears in embeddings, ignore threshold'),
                    ]
                }),
                Argument('nwords', default=-1, doc='max number of words in vocab'),
            ]
        }),
        Argument('target_map_file', doc='predefined target map file', children={
            lambda value: not value: [
                Argument('target_type', default='text', validate=lambda value: value in ('index', 'text'),
                         doc='For index type, targets are already encoded as indices like 0, 1, ...', children={
                        lambda value: value == 'text': [
                            Argument('special_targets', default=[], type=List[str],
                                     post_process=lambda val: [x.lower() for x in val],
                                     doc='specital targets that will always on top'),
                        ]
                    })]
        }),
        Argument('pretrained_embeddings',
                 validate=lambda value: not value or (
                     validate_url(value) if value.startswith('http') else io.exists(value))
                 ),
        Argument('check_max_len', default=False,
                 doc='whether to check max-len, enable this when debugging a new processor'),
        Argument('log_file', default='log.txt', type=Optional[str], doc='log filename under "data_dir"'),
        Argument('workers', default=1, validate=lambda value: value > 0),
        Argument('save_format', default='torch', validate=lambda value: value in ('torch', 'jsonl')),
        Argument('seed', default=1, type=int, doc='seed for non-pretrained embedding initialization, etc.'),
        domain='preprocess',
    )
    options.register(
        Argument(
            'processor', required=True,
            validate=lambda value: (value in processors.keys(), f'Unknown processor {value}'),
            register=lambda value: processors[value].register
        ),
        domain='processor'
    )
    options.register(
        Argument('loader', required=True, validate=lambda value: (value in loaders.keys(), f'Unknown loader {value}'),
                 register=lambda value: loaders[value].register),
        domain='loader',
    )

    def entry(args):
        arguments = options.tree(args)
        io.makedirs(args.data_dir, exist_ok=True)
        with io.open(os.path.join(args.data_dir, 'args.py'), 'w') as f:
            f.write(arguments)
        main(args)

    bootstrap(options, entry, __file__, argv)


def main(args):
    vocab_path = os.path.join(args.data_dir, 'vocab.txt')
    target_map_path = os.path.join(args.data_dir, 'target_map.txt')
    pretrained_embeddings = getattr(args, 'pretrained_embeddings', None)
    if pretrained_embeddings and pretrained_embeddings.startswith('http'):
        pretrained_embeddings = download_from_url(pretrained_embeddings)
    pretrained_embedding_path = os.path.join(args.data_dir, 'embeddings.pt')
    meta_path = os.path.join(args.data_dir, 'meta.hjson')
    log_to_file(os.path.join(args.data_dir, args.log_file))
    files = get_data_files(args)
    assert len(files) > 0, f'no files match pattern "{args.data_files}" in "{args.data_source}"'

    loader = loaders[args.loader](args)
    processor = processors[args.processor](args)
    datasets = {}
    args.__datasets__ = datasets
    resource_paths = [os.path.join(args.data_dir, path) for path in processor.resources]

    # backup previous resource files if exist
    for path in resource_paths:
        if io.exists(path):
            io.move(path, path + '.tmp')

    try:
        if args.tokenizer == 'auto':
            Dictionary.from_pretrained(args.vocab_file).save(vocab_path)
        elif args.vocab_file:
            processor.load_vocab(args.vocab_file)
            io.copy(args.vocab_file, vocab_path)
        else:
            if not datasets:
                for file in files:
                    datasets[file] = loader.load_data(file)
            dictionary = Dictionary(pad=args.pad_word, unk=args.unk_word)
            for file in files:
                data = datasets[file]
                for token in processor.token_stream(data):
                    dictionary.add_symbol(token)
            if pretrained_embeddings:
                pretrained = processor.load_embedding_dict(pretrained_embeddings)
            else:
                pretrained = []
            dictionary.finalize(threshold=args.threshold, nwords=args.nwords, pretrained=pretrained,
                                ignore_in_emb=getattr(args, 'ignore_in_emb', True))
            dictionary.save(vocab_path)
        if args.target_map_file:
            target_map = processor.load_target_map(args.target_map_file)
            io.copy(args.target_map_file, target_map_path)
        else:
            if not datasets:
                for file in files:
                    datasets[file] = loader.load_data(file)
            if args.target_type == 'index':
                target_map = Dictionary()
                symbols = set()
                for file in files:
                    data = datasets[file]
                    symbols.update(set(processor.target_stream(data)))
                symbols = sorted(symbols)
                for symbol in symbols:
                    target_map.add_symbol(symbol)
                assert len(target_map) == max(map(int, symbols)) + 1
            else:
                target_map = Dictionary(extra_special_symbols=args.special_targets)
                special_occurs = [False for _ in range(len(args.special_targets))]
                for file in files:
                    data = datasets[file]
                    for target in processor.target_stream(data):
                        target_map.add_symbol(target.lower())
                        try:
                            special_occurs[args.special_targets.index(target)] = True
                        except ValueError:
                            pass
                target_map.finalize()
                missing_special = [target for target, occurs in zip(args.special_targets, special_occurs) if not occurs]
                if missing_special:
                    raise RuntimeError('special targets not found in data: ' + str(missing_special))
            target_map.save(target_map_path)

        meta = {
            'processor': args.processor,
            'max_len': -1 if args.check_max_len else args.max_len,
            'data_size': {},
            '__version__': xdpx.__version__,
        }
        if pretrained_embeddings:
            embedding_dim = processor.extract_embeddings(pretrained_embeddings, pretrained_embedding_path)
            meta.update({'embedding_dim': embedding_dim})
        meta.update({key: val for key, val in processor.meta().items() if hasattr(args, key)})
        meta.update(
            {key: val for key, val in loader.meta().items() if hasattr(args, key) and key in processor.arguments()})
        for key, val in meta.items():
            setattr(args, key, val)

        if io.exists(meta_path):
            with io.open(meta_path) as f:
                ignored_meta = {'max_len', 'data_size', '__version__'}
                prev_meta = hjson.load(f, object_pairs_hook=dict)
                meta_keys = set(meta.keys()) - ignored_meta
                prev_meta_keys = set(prev_meta.keys()) - ignored_meta
                try:
                    unexpected_keys = meta_keys - prev_meta_keys
                    missing_keys = prev_meta_keys - meta_keys
                    assert not unexpected_keys, f'unexpected keys: {str(unexpected_keys)}'
                    assert not missing_keys, f'missing keys: {str(missing_keys)}'
                    for key in meta_keys:
                        assert prev_meta[key] == meta[key], f'({key}: {prev_meta[key]} -> {meta[key]})'
                except AssertionError as e:
                    raise RuntimeError(
                        f'Meta data have changed. Consider removing data_dir ("{args.data_dir}") and run again. ' + str(
                            e))
                prev_meta['max_len'] = max(prev_meta['max_len'], meta['max_len'])
                meta = prev_meta
    except (KeyboardInterrupt, Exception):
        # restore previous resource files
        for path in resource_paths:
            tmp_path = path + '.tmp'
            if io.exists(tmp_path):
                io.move(tmp_path, path)
        raise
    # remove backups after checking meta match
    for path in resource_paths:
        tmp_path = path + '.tmp'
        if io.exists(tmp_path):
            io.remove(tmp_path)

    activated_targets = {}
    for i, file in enumerate(files):
        data = datasets.get(file, loader.load_data(file))
        data_name = files[file]
        data = processor.numerize_samples(data)
        activated_targets[data_name] = processor.inspect(data, name=data_name)
        if args.check_max_len:
            max_len = max(processor.text_length(sample) for sample in data)
            if max_len > meta['max_len']:
                meta['max_len'] = max_len
            assert meta[
                       'max_len'] <= args.max_len, f"actual max_len {meta['max_len']} should not be larger than max_len in args {args.max_len}"
        meta['data_size'][data_name] = len(data)
        print(f'saving {data_name} ... ', end='', flush=True)
        tik = datetime.now()
        if args.save_format == 'torch':
            with io.open(os.path.join(args.data_dir, data_name + '.pt'), 'wb') as f:
                torch.save(data, f)
        elif args.save_format == 'jsonl':
            with io.open(os.path.join(args.data_dir, data_name + '.jsonl'), 'w') as f:
                for item in data:
                    print(json.dumps(item), file=f)
        else:
            raise NotImplementedError('Only support torch and jsonl now')
        print(f'''file saved ({str(datetime.now() - tik).split('.')[0]})''')
        with io.open(meta_path, 'w') as f:
            hjson.dump(meta, f)
        if i == len(files) - 1 and hasattr(processor, 'sanity_check'):
            # set args to provide the environment for sanity checks
            for name, value in meta.items():
                setattr(args, name, value)
            processor.set_epoch(0)
            print(f'| *** Example from {data_name} ***')
            for result in processor.sanity_check(random.Random(args.seed).sample(data, 5)):
                for key, value in result.items():
                    print(f'| {key}: {value}')
                print()
    for name, targets in activated_targets.items():
        if targets:
            missing = target_map.indices.keys() - targets.keys()
            if missing:
                print(f'| WARNING: these targets are missing from "{name}":\n' + ' '.join(missing))


def get_data_files(args):
    data_files = args.data_files
    if isinstance(data_files, str):
        data_files = [data_files]
    if isinstance(data_files, list):
        assert len(data_files) == len(set(data_files))
        data_files = {
            file: None for file in data_files
        }
    else:
        assert isinstance(data_files, dict)
        assert len(data_files) == len(set(data_files.values()))
    results = []
    for data_file, save_name in data_files.items():
        if save_name is not None and save_name.endswith('.pt'):
            save_name = save_name[:-3]
        path = os.path.join(args.data_source, data_file)
        root, pattern = os.path.split(path)
        assert pattern
        prev_count = len(results)
        for file in io.listdir(root):
            if fnmatch(file, pattern):
                results.append((
                    os.path.join(root, file),
                    save_name or os.path.splitext(file)[0].replace('/', '_'),
                ))
        if len(results) == prev_count:
            raise IOError(f'"{data_file}" does not match any file in "{args.data_source}"')
    return OrderedDict(sorted(results))


if __name__ == '__main__':
    cli_main()
