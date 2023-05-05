import os
import re
import sys
import math
import time
import atexit
import shlex
import hjson
import getpass
import importlib
import subprocess
import contextlib
import threading
import traceback
import torch
import random
import hashlib
from fnmatch import fnmatch
from tabulate import tabulate
from io import StringIO
from datetime import datetime, date
from typing import List, Optional
from collections import deque
from tqdm import tqdm
import numpy as np
import pandas as pd
from urllib.parse import quote
from requests_html import HTMLSession
from .io_utils import IO, DefaultIO, OSS
from .distributed_utils import worker_master_first, is_worker_master, should_barrier


class _IOWrapper:
    def __init__(self):
        self._io = DefaultIO()

    def set_io(self, new_io):
        self._io = new_io

    def __getattr__(self, name):
        if hasattr(self._io, name):
            return getattr(self._io, name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            raise AttributeError(f"'io' object has no attribute '{name}'")

    def __str__(self):
        return self._io.__name__


def import_user_module(module_root=None, reload=False):
    from glob import glob
    if module_root is None:
        import xdpx
        xdpx_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        module_root = os.path.join(xdpx_root, 'user/modules')
    module_paths = glob(os.path.join(module_root, '[!_]*.py'))
    modules = {}
    for module_path in module_paths:
        module_path = os.path.abspath(module_path)
        module_parent, module_name = os.path.split(module_path)
        module_name = os.path.splitext(module_name)[0]

        if module_name not in sys.modules or reload:
            sys.path.insert(0, module_parent)
            importlib.import_module(module_name)
            module = sys.modules[module_name]
            modules[module_name] = module
            if reload:
                importlib.reload(module)
            sys.path.pop(0)
    return modules


io: IO = _IOWrapper()
import_user_module()


def register(name=None, registry=None):
    def decorator(fn, registration_name=None):
        module_name = registration_name or _default_name(fn)
        if module_name in registry:
            raise LookupError(f"module {module_name} already registered.")
        registry[module_name] = fn
        return fn

    return lambda fn: decorator(fn, name)


def _default_name(obj_class):
    return obj_class.__name__


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def cast_to_half(sample):
    def _cast_to_half(tensor):
        if tensor.dtype is torch.float32:
            return tensor.half()
        return tensor

    return apply_to_sample(_cast_to_half, sample)


def move_to_cpu(sample):
    def _move_to_cpu(tensor):
        return tensor.cpu()

    return apply_to_sample(_move_to_cpu, sample)


def pin_memory(sample):
    def _pin_memory(tensor):
        return tensor.pin_memory()

    return apply_to_sample(_pin_memory, sample)


def convert_to_native(sample):
    def _convert_to_native(tensor):
        return tensor.item() if tensor.numel() == 1 else tensor.cpu().tolist()

    return apply_to_sample(_convert_to_native, sample)


def half_precision(sample):
    def _apply_half(t):
        if t.dtype is torch.float32:
            return t.half()
        return t

    return apply_to_sample(_apply_half, sample)


def current_time(for_path=False) -> str:
    if for_path:
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


log_cache = deque()


def log_to_file(log_path, prefix='| '):
    # redirect stdout also to log file
    import builtins as __builtin__
    builtin_print = __builtin__.print
    io.makedirs(os.path.dirname(log_path), exist_ok=True)

    def cached_print(*args, **kwargs):
        if args and str(args[0]).startswith(prefix):
            default_stdout = sys.stdout
            cache = StringIO()
            sys.stdout = cache
            builtin_print(current_time(), *args, **{'flush': True, **kwargs})
            value = cache.getvalue()
            if value:
                log_cache.append((log_path, value))
                if len(log_cache) == 1:
                    thread = threading.Thread(target=delayed_flush)
                    thread.start()
                default_stdout.write(value)
                default_stdout.flush()
            sys.stdout = default_stdout
        else:
            builtin_print(*args, **kwargs)

    __builtin__.print = cached_print


def delayed_flush(delay=1):
    time.sleep(delay)
    logs = {}
    while log_cache:
        log_path, value = log_cache.popleft()
        if log_path not in logs:
            logs[log_path] = []
        logs[log_path].append(value)
    for log_path, values in logs.items():
        with io.open(log_path, 'a') as f:
            for value in values:
                f.write(value)


def load_module(path: str):
    from importlib.util import spec_from_file_location, module_from_spec
    assert path.endswith('.py')
    spec = spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_script(name):
    import xdpx
    return load_module(os.path.join(os.path.dirname(xdpx.__path__[0]), f'scripts/{name}.py'))


def run_script(name: str, cli_args: List[str]):
    module = load_script(name)
    return module.cli_main([name] + cli_args)


def run_script_cli(argv=sys.argv):
    if len(argv) == 1:
        print('Usage: x-script <script_name> <script_arg1> <script_arg2> ...')
        exit()
    script_name = argv.pop(1)
    return run_script(script_name, argv[1:])


def diff_params(args1, args2, exclude=['save_dir']):
    if not isinstance(args1, dict):
        args1 = vars(args1)
    if not isinstance(args2, dict):
        args2 = vars(args2)
    exclude = set(exclude)
    diff = []
    for key1, val1 in args1.items():
        if key1 in exclude or key1.startswith('__'):
            continue
        if key1 not in args2:
            diff.append(f'-- {key1}: {val1}')
        elif val1 != args2[key1]:
            diff.append(f'+- {key1}: {val1} -> {args2[key1]}')
    for key2, val2 in args2.items():
        if key2 in exclude or key2.startswith('__'):
            continue
        if key2 not in args1:
            diff.append(f'++ {key2}: {val2}')
    diff.sort()
    return diff


def should_save_meta(args):
    from xdpx.utils.distributed_utils import is_master
    return is_master(args) and hasattr(args, '__cmd__') and args.__cmd__ == 'train' and io.isdir(args.save_dir)


def get_train_subsets(args, reload=False) -> List[str]:
    if reload:
        with io.open(os.path.join(args.data_dir, 'meta.hjson')) as f:
            meta = hjson.load(f)
            data_list = meta['data_size'].keys()
    else:
        data_list = args.data_size.keys()

    train_subsets = args.train_subset
    if isinstance(args.train_subset, str):
        train_subsets = [train_subsets]
    train_subsets = [subset for subset in data_list
                     if any(fnmatch(subset, train_subset) for train_subset in train_subsets)]
    if getattr(args, 'exclude_valid_from_train', False):
        train_subsets = [subset for subset in train_subsets if subset != args.valid_subset]
    return train_subsets


def parse_model_path(path: str, args):
    if path is None:
        return path
    if path.endswith('<best>') or path.endswith('<last>'):
        from xdpx.trainer import Trainer
        if path.startswith('<'):
            assert args is not None
            assert args.save_dir
            save_dir = args.save_dir
        else:
            save_dir = os.path.dirname(path)
        if path.endswith('<best>'):
            try:
                with io.open(os.path.join(save_dir, 'valid.log.tsv')) as f:
                    df = pd.read_csv(f, sep='\t', header=0)
                    if df['best_score'].iloc[-1] >= df['best_score'].iloc[0]:
                        step = int(df.iloc[df['best_score'].idxmax()]['step'])
                    else:
                        step = int(df.iloc[df['best_score'].idxmin()]['step'])
            except FileNotFoundError:
                raise ValueError('Cannot parse <best> without log files.')
        else:  # endswith('<last>')
            steps = []
            for model_path in io.listdir(save_dir, contains=Trainer.save_prefix):
                step = Trainer.get_step_from_path(model_path)
                steps.append(step)
            if not steps:
                raise FileNotFoundError(path)
            step = max(steps)
        return os.path.join(save_dir, Trainer.save_pattern(step))
    return path


def parse_relative_config(val, args):
    if not val:
        return val
    if not val.endswith('.hjson'):
        val += '.hjson'
    if val.startswith('.'):
        from xdpx.options import Options
        val = Options.parse_relative_path(val, os.path.dirname(args.__origin__))
    return val


def download_from_url(url):
    import requests
    with worker_master_first():
        cache = cache_file(url, dry=True)
        if os.path.exists(cache):
            print('Use existed cache for', url)
            return cache
        print('Download from', url)
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        try:
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc='Downloading') as pbar, \
                    open(cache, 'wb') as f:
                for data in r.iter_content(chunk_size=32 * 1024):
                    f.write(data)
                    pbar.update(len(data))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(cache):
                os.remove(cache)
            raise e
    return cache


def validate_url(url):
    import urllib
    cache = cache_file(url, dry=True)
    if os.path.exists(cache):
        # will use existed cache
        return True
    try:
        with urllib.request.urlopen(url):
            return True
    except Exception:
        print(traceback.format_exc())
        return False


def load_args(path):
    from xdpx.options import Options
    try:
        args = Options.load_hjson(os.path.join(path, 'args.hjson'))
    except FileNotFoundError:
        with io.open(os.path.join(path, 'args.py')) as f:
            args = vars(Options.parse_tree(eval(f.read())))
    return args


def has_parameters(module):
    try:
        next(module.parameters())
        return True
    except StopIteration:
        return False


def compress_dir(src: str, dst: Optional[str] = None, exclude=[], max_size=None, errmsg=None):
    from zipfile import ZipFile
    paths = []

    assert io.isdir(src)
    for root, dirs, files in os.walk(src):
        dirs[:] = [dirname for dirname in dirs if not dirname.startswith(('__', '.', 'user'))]
        for filename in files:
            if not any(keyword in filename for keyword in exclude):
                paths.append(os.path.join(root, filename))
    if not dst:
        dst = src.rstrip('/')
    if not dst.endswith('.zip'):
        dst += '.zip'
    prefix_len = len(src.rstrip('/')) + 1
    dst_cache = cache_file(dst, dry=True)
    with ZipFile(dst_cache, 'w') as pack:
        for path in paths:
            pack.write(path, arcname=path[prefix_len:])
    if max_size is not None:
        if io.size(dst_cache) > max_size:
            raise IOError(errmsg or f'File size exceeds limit: {max_size}')
    md5 = io.md5(dst_cache)
    io.move(dst_cache, dst)
    return md5


def shell_command(command: str) -> str:
    import subprocess
    import shlex
    result = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, encoding='utf-8')
    return result.stdout


def get_commit_hash():
    if os.path.exists('.git_version'):
        with open('.git_version') as f:
            commit = f.readlines()[0].rstrip()
    else:
        commit = subprocess.run(shlex.split('git rev-parse HEAD'), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, encoding='utf-8').stdout.rstrip()
        if not commit:
            raise RuntimeError('Cannot get commit hash')
    return commit


def is_apex_available():
    if torch.cuda.is_available():
        try:
            from apex import amp
            from apex.parallel import DistributedDataParallel
            from apex.normalization import FusedLayerNorm
            return True
        except (ImportError, ModuleNotFoundError):
            return False
    return False


def persistent_run(func, *args, **kwargs):
    for i in range(3):
        try:
            return func(*args, **kwargs)
        except Exception:
            if i == 2:
                print(traceback.format_exc())


def get_total_steps(args, runtime, calibrate=True):
    total_steps = math.inf
    if args.max_epoch:
        from xdpx.trainer import Trainer
        actual_batch_size = args.batch_size * (args.distributed_world_size if runtime else 1) * args.update_freq
        shard_data = [args.data_size[subset] for subset in get_train_subsets(args)]

        def dist_calibrate(num):
            if not calibrate:
                return num
            return int(math.ceil(num / args.distributed_world_size)) * args.distributed_world_size

        if not args.lazy_load:
            total_steps = dist_calibrate(sum(shard_data)) // actual_batch_size * args.max_epoch
        else:
            total_steps = sum(
                ([dist_calibrate(num) // actual_batch_size for num in shard_data]
                 * (args.max_epoch // len(shard_data) + 1)
                 )[:args.max_epoch]
            )
    if args.max_update:
        total_steps = min(args.max_update, total_steps)
    assert total_steps is not math.inf
    return total_steps


default_cache_root = f'/tmp/{getpass.getuser()}/filecache/'

default_cache_root = os.environ.get('XDPX_CACHE_PATH', default_cache_root)


def cache_file(file, root=default_cache_root, clear_cache=False, dry=False):
    """
    Cache file to local disks with auto distributed barrier.
    DO NOT USE this function in single-node multi-processing.
    """
    if os.path.exists(file) or (io.islocal(file) and dry):
        return file

    cache = os.path.join(root, re.sub(r'(oss|https?)://', '', file).strip('/'))
    parent = os.path.dirname(cache)
    try:
        os.makedirs(parent, exist_ok=True)
    except FileExistsError:
        os.remove(parent)
        os.makedirs(parent)

    def clear_cache_callback():
        if io.exists(cache):
            io.remove(cache)

    if dry:
        if clear_cache:
            atexit.register(clear_cache_callback)
        return cache

    is_master = is_worker_master()

    def download_cache():
        try:
            io.copy(file, cache)
        except (Exception, KeyboardInterrupt) as e:
            if io.exists(cache):
                io.remove(cache)  # avoid corrupted files to be loaded by the following runs
            raise e
        with open(meta_path, 'w') as f:
            f.write(io.last_modified_str(file))
        if clear_cache:
            atexit.register(clear_cache_callback)

    if is_master:
        meta_path = cache + '.last-modified'
        if not io.exists(cache):
            download_cache()
        else:
            waiting = 0
            while (datetime.now() - io.last_modified(cache)).total_seconds() < 1:
                # another process is writing to the file
                waiting += 1
                print(f'\rWaiting for another process to finish downloading..({waiting * 5}s)', end='')
                time.sleep(5)
            if waiting:
                print()
            if not os.path.exists(meta_path):
                # last download does not complete
                print('Cache file was broken. Download again.')
                download_cache()
            else:
                with open(meta_path) as f:
                    last_modified = f.read().strip()
                online_lm = io.last_modified_str(file)
                if online_lm != last_modified:
                    print(f'Warning: OSS file has changed. Download again for {file}')
                    download_cache()

    if should_barrier():
        torch.distributed.barrier()

    return cache


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextlib.contextmanager
def torch_seed(seed, cuda=torch.cuda.is_available()):
    assert isinstance(seed, int)
    rng_state = torch.get_rng_state()
    if cuda:
        cuda_rng_state = torch.cuda.get_rng_state()
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    yield
    torch.set_rng_state(rng_state)
    if cuda:
        torch.cuda.set_rng_state(cuda_rng_state)


def is_chinese(text, strict=True):
    """
    Reference: https://github.com/google-research/bert/blob/master/tokenization.py#L264
    """
    buffer = []
    for char in text:
        code = ord(char)
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        buffer.append((
                (0x4E00 <= code <= 0x9FFF)
                or (0x3400 <= code <= 0x4DBF)
                or (0x20000 <= code <= 0x2A6DF)
                or (0x2A700 <= code <= 0x2B73F)
                or (0x2B740 <= code <= 0x2B81F)
                or (0x2B820 <= code <= 0x2CEAF)
                or (0xF900 <= code <= 0xFAFF)
                or (0x2F800 <= code <= 0x2FA1F)
        ))
    return all(buffer) if strict else any(buffer)


def pformat_dataframe(dataframe, showindex=False, **kwargs):
    return tabulate(dataframe, tablefmt='psql', headers='keys', showindex=showindex, **kwargs)


def format_time_span(start: float, end: float) -> str:
    ft = time.strftime("%H:%M:%S", time.gmtime(end - start))
    days = (date.fromtimestamp(end) - date.fromtimestamp(start)).days
    if days == 1:
        ft = f'{days} day {ft}'
    elif days > 1:
        ft = f'{days} days {ft}'
    return ft


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        if self.decay >= 1.0:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        if self.decay >= 1.0:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        if self.decay >= 1.0:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        if self.decay >= 1.0:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


from googlesearch import get_page, quote_plus
from lxml import etree
import requests
import json

DEFAULT_SESSION = requests.Session()
DEFAULT_SESSION.cookies = requests.utils.cookiejar_from_dict(
    {'sm_uuid': '45f5c34c3719d39b4003f636b8f5a72e%7C%7C%7C1653558899'}, cookiejar=None, overwrite=True)


def wrap_shenma_snippet(sc_name, snippet, url=None):
    if snippet:
        if not isinstance(snippet, str):
            print(f'warning: {snippet} not string , ignored')
            return None
        snippet = snippet.strip().replace('<em>', '').replace('</em>', '').replace('...', '')
        if snippet:
            return {'sc_name': sc_name, 'snippet': snippet, 'url': url}
    return None


question_regex = [
    re.compile(
        r'(？|\?|吗|什么|怎么|怎样|咋|啥|如何|为什么|哪|几|谁|多少|多大|多高|是不是|有没有|是否|多久|可不可以|能不能|行不行)'),
    re.compile('(是).+(还是)')]
persona_regex = [
    re.compile(
        '(我|你)+(是|是不是|也是|叫啥|叫什么|几岁|多少岁|毕业|多大|哪里|经常|一般|平时|平常|谁|会|还会|工作|名字|姓名|小名|大名|全名|年龄|年纪|工作|职业|干什么)+'),
    re.compile('(我的|你的)+(名字|姓名|昵称|名称|全名|大名|年纪|年龄|工作|职业|学校|宠物|猫|狗|爱好|大学)+'),
    re.compile('(我|你)+(的)*(父母|爸|妈|男朋友|女朋友|哥|姐|妹|弟|老公|老婆|孩子|女儿|儿子)+'),
    re.compile('(我|你)+(是)*(男的|女的|男生|女生|男孩|女孩|性别)+')
]


def text_is_question(query):
    q = query.replace(' ', '')
    for r in question_regex:
        if r.findall(q):
            return True
    return False


def search_shenma(query, test_url=False):
    # url_search = "https://agg.sm.cn/api/s_api?q={}&ft=json&no_html=1&ad=no&osr=damoyuan&sl=31&uid=45f5c34c3719d39b4003f636b8f5a72e%7C%7C%7C1653558899"
    if test_url:
        url_search = "https://test.m.sm.cn/api/s_api?q={}&ft=json&no_html=1&ad=no&osr=damoyuan&sl=23&uid=45f5c34c3719d39b4003f636b8f5a72e%7C%7C%7C1653558899"
    else:
        url_search = "https://agg.sm.cn/api/s_api?q={}&ft=json&no_html=1&ad=no&osr=damoyuan&sl=23&uid=45f5c34c3719d39b4003f636b8f5a72e%7C%7C%7C1653558899"
    query = quote_plus(query)
    url = url_search.format(query)
    # text = get_page(url)
    text = DEFAULT_SESSION.get(url).text
    try:
        data = json.loads(text)
    except Exception as e:
        print(f'parse json {url} error')
        print(e)
        print(text)
        data = {}

    items = data.get('items', {}).get('item', [])
    snippets = []

    for item in items:
        title = item.get('title', '')
        if title and '视频' in title:
            continue
        sc_name = item.get('sc_name', '')
        if sc_name in ('news_natural', 'structure_web_bbs', 'text_recommend', 'short_video',
                       'structure_short_video', 'xiami_lyric_song', 'kg_recommend_n', 'structure_doc',
                       'kg_recommend_dim_1',
                       'medical_hq_video', 'kg_recommend_n', 'doc_sc_0', 'doc_sc_1', 'doc_sc_3'
                       ):
            continue
        # desc = item.get('desc', '')
        # snippet = f'{title} {desc}'.strip()
        # snippets.append(wrap_shenma_snippet(sc_name, snippet))
        url = item.get('url', '')
        snippets.append(wrap_shenma_snippet(sc_name, item.get('desc'), url))
        try:
            if sc_name:
                data = item.get(sc_name)
                if sc_name == 'weather_new_huake':
                    city = data.get('item', {}).get('city', '')
                    for i in range(1, 2):  # only today
                        dkey = f'day_{i}'
                        it = data.get('item', {}).get(dkey, {})
                        dname = it.get('week_day', '')
                        weather2 = it.get('weather2', '')
                        temp = it.get('temp', '')
                        windPowerLevel = it.get('windPowerLevel', '')
                        windDirection = it.get('windDirection', '')
                        snippet = f'{dname}{city}天气{weather2} 气温{temp} {windDirection}{windPowerLevel}'
                        snippets.append(wrap_shenma_snippet(sc_name, snippet, url))
                    break
                elif sc_name == 'weather_moji':
                    city = data.get('wisdomData', {}).get('city', '')
                    break
                elif sc_name == 'finance_stock_new':
                    name = data.get('Name', '')
                    moduleData = data.get('moduleData', {})
                    gc = {d['label']: d['value'] for d in moduleData.get('grid_container', {})}
                    zuidi = gc.get('最低', '')
                    zuigao = gc.get('最高', '')
                    zuoshou = gc.get('昨收', '')
                    jinkai = gc.get('今开', '')
                    shizhi = gc.get('市值', '')
                    snippet = f'{name}股价, 昨天收盘价{zuoshou}元, 今日开盘价{jinkai}元'
                    snippets.append(wrap_shenma_snippet(sc_name, snippet, url))
                elif sc_name == 'wenda_selected':
                    snippet = data.get('item', {}).get('name', '')
                    answer = data.get('item', {}).get('answer', {})
                    if isinstance(answer, str):
                        snippet += answer
                    else:
                        answer = answer.get('item', '')
                        if isinstance(answer, list):
                            snippet += ' '.join(answer)
                        elif isinstance(answer, str):
                            snippet += answer
                    snippets.append(wrap_shenma_snippet(sc_name, snippet, url))
                elif sc_name == 'structure_web_info':
                    if data.get('time') > '2022':
                        snippet = item.get('recoTitle', '')
                        snippets.append(wrap_shenma_snippet(sc_name, snippet, url))
                elif sc_name == 'structure_web_how':
                    snippet = ''
                    for k in ('SP_HOW_STEP_FIRST', 'SP_HOW_STEP_SECOND', 'SP_HOW_STEP_THIRD', 'SP_HOW_STEP_FOURTH'):
                        snippet += data.get(k, '')
                    snippets.append(wrap_shenma_snippet(sc_name, snippet, url))
                elif sc_name == 'yisou_film':
                    film_name = data.get('name', '')
                    brief = data.get('brief', '')
                    directors = data.get('directors', [])
                    actors = data.get('actors', [])
                    snippet = film_name + brief
                    if directors:
                        snippet += '导演是' + '、'.join(directors)
                    if actors:
                        snippet += '主演是' + '、'.join(actors)
                    snippets.append(wrap_shenma_snippet(sc_name, snippet, url))
                elif sc_name == 'baike_sc':
                    name = data.get('name', '')
                    abstract = data.get('abstract', '')
                    if abstract:
                        snippets.append(wrap_shenma_snippet(sc_name, abstract, url))
                    else:
                        text = item.get('moduleData', {}).get('baike_info', '')
                        url = item.get('moduleData', {}).get('baike_url', '')
                        snippets.append(wrap_shenma_snippet(sc_name, text, url))
                    basic = data.get('basic', [])

                    if basic:
                        kv_info = ''
                        for kv in basic:
                            key = kv.get('key')
                            value = kv.get('value')
                            kv_info += f'{name} {key} {value} </s>'
                        snippets.append(wrap_shenma_snippet(sc_name, kv_info, url))
                elif sc_name == 'peoplestarzeus':
                    kg_data = data.get('kg_data', {})
                    name = kg_data.get('name', '')
                    notable_for = kg_data.get('notable_for', '')
                    date_of_birth_with_age = kg_data.get('date_of_birth_with_age', '')
                    rel_person = kg_data.get('rel_person', {}).get('item', [])
                    snippet = f'{name} {notable_for} 出生日期 {date_of_birth_with_age}</s>'
                    for desc_name in rel_person:
                        name2 = desc_name.get('name', '')
                        desc = desc_name.get('desc', '')
                        snippet += f'{name} {desc} {name2}</s>'
                    snippets.append(wrap_shenma_snippet(sc_name, snippet, url))
                elif sc_name == 'kk_kg_entity_people':
                    snippet = data.get('sense_name', ' ') + data.get('abstract', ' ')
                    name = data.get('name')
                    baike_kv = data.get('baike_kv', {}).get('item', [])
                    if name and baike_kv:
                        triples = ['{} {} {}'.format(name, d.get('label', ''), d.get('value', '')) for d in baike_kv]
                        snippet += '</s>'.join(triples)
                    snippets.append(wrap_shenma_snippet(sc_name, snippet, url))
                elif sc_name == 'news_uchq':
                    display = data.get('display', {}).get('summary', '')
                    source = data.get('display', {}).get('source', '')
                    if display:
                        snippet = f'{source}消息: {display}'
                        snippets.append(wrap_shenma_snippet(sc_name, snippet, url))
                    news_node = data.get('news_node', [])
                    for node in news_node[:3]:
                        time = node.get("time", '')
                        if '分钟前' in time or '小时前' in time:
                            snippet = '{} {}'.format(node.get('title'), node.get('summary'))
                            snippets.append(wrap_shenma_snippet(sc_name, snippet, url))
                elif sc_name == 'news_top_list':
                    news = data.get('display', {}).get('list_info', {}).get('fields', [])
                    for new in news[:20]:
                        if type(new) == dict:
                            title = new.get('title', '')
                            summary = new.get('news_summary', '')
                            snippet = f'头条新闻: {title};;;新闻摘要: {summary}'
                            snippets.append(wrap_shenma_snippet(sc_name, snippet, url))

                elif sc_name == 'covid_19':
                    try:
                        tab_container = data.get('wisdomData', {}).get('tab_container', [])

                        if tab_container and type(tab_container[0]) == dict:
                            data_new = tab_container[0].get('data_new', '[]')
                            data_new = json.loads(data_new)
                            text_new = data.get('wisdomData', {}).get('text_new', '')
                            snippet = f'{text_new},'
                            for d in data_new:
                                title = d.get('title', '')
                                val = d.get('val', '')
                                snippet += f'{title}{val}, '
                            snippets.append(wrap_shenma_snippet(sc_name, snippet, url))
                            break
                    except:
                        print(f'warning: covid_19 card parse error')
        except:
            print(f'warning parse error')

    snippets = [t for t in snippets if
                t is not None and t.get('snippet') is not None]
    # sc_names = set([t.get('sc_name') for t in snippets])
    # if 'wenda_selected' in sc_names or 'baike' in sc_names or 'baike_sc' in sc_names or 'structure_qna' in sc_names:
    #     snippets = [t for t in snippets if t.get('sc_name') != 'structure_web_info']
    return snippets


def search_shenma_html(query):
    url_search = "https://m.sm.cn/s?q={}&from=smor&safe=1&by=submit&snum=30"
    query = quote_plus(query)
    url = url_search.format(query)
    html = DEFAULT_SESSION.get(url).text
    # html = get_page(url_search.format(query))
    html = etree.HTML(html)
    html_data = html.xpath('//div[@id="results"]/div')
    return_data = []
    for div in html_data:
        if 'id' in div.attrib:
            id_str = div.attrib['id']
            if id_str.startswith('ad_'):  # 广告
                continue
            elif id_str.startswith('Sc_Shopping'):
                continue

        url = div.xpath('.//a/@href')
        data = div.xpath('.//*[contains(@c-bind, "data.text")]')

        if data and url:
            url = url[0]
            content = '\t[SEP]\t'.join([d.xpath('string(.)') for d in data[1:]])
            if '相关视频' in content or not url.startswith('http') or len(content) < 8:
                continue

            title = div.xpath('.//div[contains(@class, "c-header-title")]')
            if title:
                title = title[0].xpath('string(.)')
            else:
                title = ''
            if '视频' in title:
                continue

            item = {
                'id': 'sm_{}'.format(len(return_data)),
                'url': url,
                'title': title,
                'snippet': content
            }
            return_data.append(item)

    return return_data


def search_google(query):
    url_search = "https://www.google.com.hk/search?q={}"
    query = quote_plus(query)
    html = get_page(url_search.format(query))
    html = etree.HTML(html)
    html_data = html.xpath('//div[contains(@class, "ezO2md")]')
    return_data = []
    for div in html_data:
        url = div.xpath('.//a/@href')
        title = div.xpath('.//span[contains(@class, "CVA68e qXLe6d")]')
        data = div.xpath('.//span[contains(@class, "qXLe6d FrIlee")]/span[@class="fYyStc"]')

        if data and url and title:
            url = url[0]
            title = title[0].xpath('string(.)')
            content = '\t[SEP]\t'.join([d.xpath('string(.)') for d in data])
            item = {
                'id': 'g_{}'.format(len(return_data)),
                'url': url,
                'title': title,
                'snippet': content
            }
            return_data.append(item)

    return return_data


def get_random_baidu_cookie():
    cookies = [
        'PSTM=1645440125; BAIDUID=49D5966BB6F2D98A8378EC10151CE748:FG=1; BAIDUID_BFESS=49D5966BB6F2D98A8378EC10151CE748:FG=1; BIDUPSID=5C48EADF0E27C74CB11F290539E5EAA8; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; __yjs_duid=1_6b058121c11c500f39afbc042ec623711645440178604; delPer=0; PSINO=7; MCITY=-257%3A; BA_HECTOR=05a0ak0ga42525a5us1h18lb30r; BDRCVFR[C0p6oIjvx-c]=rJZwba6_rOCfAF9pywd; H_PS_PSSID=35105_35865_34584_35491_35872_35246_35319; ab_sr=1.0.1_ZGM2MTQ3YjE2NGE0ZmE2NWNhNGYzMDQ1Nzg1ZWYxYWFjZDllZjA1NzY0YWE3NjVjZmEyNjA4NmE5NTljZTEzOTFkNzViMWRlNTA4ZmQwYWIzYWZlYjQyMDYxZTcxNGI0NWVjYzU5ODk0ZDVmYmNkZDI4YzkyNGEwNTUwZjc4MWU3Y2Q0ZTUzOGExNjQwZTgzMzM4ZjQ2ZjkzMjE0OGNjZA==; BAIDU_WISE_UID=wapp_1645499858512_985',
        'BIDUPSID=0AB15879656FD166028DF65039BDFF15; PSTM=1641442191; BAIDUID=911EF71E90573B2693EC612910B1F7BE:FG=1; BCLID_BFESS=9239639223377566883; BDSFRCVID_BFESS=1T-OJeCmHxdstirHc7RXbo9jumKK0gOTHllnPXllHP8_1buVJeC6EG0Ptf8g0KubFTPRogKK0gOTH6KF_2uxOjjg8UtVJeC6EG0Ptf8g0M5; H_BDCLCKID_SF_BFESS=tJkD_I_hJKt3fP36q6_a2-F_2xQ0etJXf5Txbp7F5lOVO-ngKU613MkSjNOj5t482jTLahkM5h7xObR1hl3ih-An0a7dJ4jtQeQ-5KQN3KJmfbL9bT3v5tDz3b3N2-biWbRM2MbdJqvP_IoG2Mn8M4bb3qOpBtQmJeTxoUJ25DnJhhCGe6-MjT3-DG8jqbvEHDc-WJ3t-TrjDCvRhMjcy4LdjG5N0PJT5bv73K022boobJcGLqjW0R_X3-Aq54RMagQwLPJEytQTS-5VbtoMQfbQ0-cOqP-jWbnu-qTo2n7JOpkRbUnxy50vQRPH-Rv92DQMVU52QqcqEIQHQT3m5-5bbN3ht6IHJbCJoDD5tIvbfP0kjjQWMt_h-fuX5-CstGPL2hcH0b61JbbR5-rKy-JW0R7a25cBbCjiaKJjBMb1DbRk0h7ShMkrebPD5JQpWDTm_q5TtUJMeCnTDMRh-xK70b5yKMnitIv9-pPKWhQrh459XP68bTkA5bjZKxtq3mkjbPbDfn028DKu-n5jHj3WDG-J3q; __yjs_duid=1_ada3d0ac8d4be7042dd53d52221555631641452261829; BAIDUID_BFESS=911EF71E90573B2693EC612910B1F7BE:FG=1; BD_HOME=1; H_PS_PSSID=35104_31660_34584_35490_35841_35887_35542_35318_26350_35867_22158; BD_UPN=12314753; delPer=0; BD_CK_SAM=1; PSINO=7; H_PS_645EC=09c89Z6QKcJ4xzJZr1LUqxrp0qdbpltyn/ixDDrfq5R6r0cQWwLiJT3HLZY; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; BA_HECTOR=a424810gag04818hg31h15uop0q; baikeVisitId=492b5e23-3a27-4d6d-bf0a-ab5907361a87; BDSVRTM=643'
    ]
    cooke = random.choice(cookies).strip()
    return cooke


def get_baidu_page(kw):
    cookie = get_random_baidu_cookie()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
        "Referer": "https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&rsv_idx=2&ch=&tn=baiduhome_pg&bar=&wd=123&oq=123&rsv_pq=896f886f000184f4&rsv_t=fdd2CqgBgjaepxfhicpCfrqeWVSXu9DOQY5WyyWqQYmsKOC%2Fl286S248elzxl%2BJhOKe2&rqlang=cn",
        # "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        "Sec-Fetch-Mode": "navigate",
        "Cookie": cookie,
        "Connection": "Keep-Alive",
    }

    if cookie:
        if "__yjs_duid" not in cookie:
            pass
        else:
            _ = cookie.split("__yjs_duid=")
            __ = _[1].split(";", 1)[-1]
            ___ = hashlib.md5()
            cookie = _[0] + "__yjs_duid=1_" + str(___.hexdigest()) + __

    headers["Cookie"] = cookie + ";random=" + str(random.randint(500, 4000))

    text = quote(kw, "utf-8")
    cp = 1
    rsv_page = '&rsv_page=1'

    url = f"https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&tn=baidu&wd={kw}&oq={text}&pn={(cp - 1) * 10}&inputT={random.randint(500, 4000)}{rsv_page}"

    session = HTMLSession()
    r = session.get(url, headers=headers)
    r.encoding = r.apparent_encoding
    content_html = r.html.html
    return content_html


def search_baidu(query):
    html = get_baidu_page(query)
    html = etree.HTML(html)

    html_data = html.xpath('//div[@id="content_left"]')
    return_data = []
    if html_data:
        div = html_data[0]
        containers = div.xpath('.//div[contains(@class, "c-container xpath-log")]')

        if containers:
            for c in containers:
                url = c.xpath('.//a/@href')
                title = c.xpath('.//h3')

                # text_2NOr6
                if url and title:
                    url = url[0]
                    title = title[0].xpath('string(.)')
                    if '百度百科' in title:
                        div = c.xpath('.//div[contains(@class,"text_2NOr6")]')
                        content = '\t'.join([d.xpath('string(.)') for d in div])
                    else:
                        span = c.xpath('.//span[contains(@class, "content-right_8Zs40")]')
                        content = '\t'.join([d.xpath('string(.)') for d in span])
                    if content:
                        item = {
                            'id': 'bd_{}'.format(len(return_data)),
                            'url': url,
                            'title': title,
                            'snippet': content
                        }
                        return_data.append(item)

    return return_data


def search_sogou(query):
    url_search = "https://www.sogou.com/web?query={}"
    query = quote_plus(query)
    html = get_page(url_search.format(query))

    html = etree.HTML(html)

    html_data = html.xpath('//div[@id="main"]')
    return_data = []
    if html_data:
        div = html_data[0]
        containers = div.xpath('.//div[contains(@class, "vrwrap")]')

        if containers:
            for c in containers:
                url = c.xpath('.//a/@href')
                title = c.xpath('.//h3')
                div = c.xpath('.//div[contains(@class, "img-flex")]')
                if url and title and div:
                    url = url[0]
                    title = title[0].xpath('string(.)')
                    content = '\t[SEP]\t'.join([d.xpath('string(.)') for d in div]).replace('\t', '').replace('\r\n',
                                                                                                              '').replace(
                        ' ', '')

                    item = {
                        'id': 'sg_{}'.format(len(return_data)),
                        'url': url,
                        'title': title,
                        'snippet': content
                    }
                    return_data.append(item)

    return return_data
