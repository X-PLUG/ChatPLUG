# from __future__ import annotations  # available since Python 3.7
import re
import os
import torch
import inspect
import hjson
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from copy import copy, deepcopy

from typeguard import check_type
from collections.abc import Iterable
from typing import List, Callable, Optional, Union, Dict, Any, Tuple


def explicit_checker(f):
    varnames = inspect.getfullargspec(f)[0]
    def wrapper(*a, **kw):                 # exclude self
        kw['_explicit'] = set(list(varnames[1: len(a)]) + list(kw.keys()))
        return f(*a, **kw)
    return wrapper


class Arg:
    """
    @DynamicAttrs
    """
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
        
    def strip_prefix(self, prefix):
        new_args = {}
        # if names collapse, stripped ones will replace original ones
        for key, val in sorted(vars(self).items(), key=lambda item: len(item[0])):
            if key.startswith(prefix) and len(key) > len(prefix):
                key = key[len(prefix):]
            new_args[key] = val
        return Arg(**new_args)

    def copy(self):
        return deepcopy(self)

    def update(self, *args):
        for arg in args:
            for key, val in vars(arg).items():
                setattr(self, key, val)
        return self

    def change(self, **kwargs):
        """this method is not in-place"""
        args = self.copy()
        for key, val in kwargs.items():
            if not isinstance(val, dict):
                setattr(args, key, val)
            else:
                getattr(args, key).update(val)
        return args

    # args that are not crucial to training results
    __exclude__ = set('__exclude__ resume overwrite save_dir distributed_rank device_id'
                      ' viz_ref_dir log_file figext data_size train_steps'.split())

    def __eq__(self, other):
        if isinstance(other, Arg):
            other = vars(other)
        compare_keys = other.keys() - self.__exclude__
        if vars(self).keys() - self.__exclude__ != compare_keys:
            return False
        for key in compare_keys:
            if getattr(self, key) != other[key]:
                return False
        return True

    def __hash__(self):
        keys = sorted(vars(self).keys() - self.__exclude__)
        return hash(''.join(str(getattr(self, key)) for key in keys))


class Argument:
    _type = type
    @explicit_checker
    def __init__(
            self, name: str, type=str, default=None, doc: Optional[str]=None, required=False,
            validate: Union[Callable[[Any], Union[bool, Tuple]], List[Callable]] = [],
            register: Optional[Callable] = None, children: Union[list, Dict[Callable, list]]=[],
            post_process: Optional[Callable[[Any, Arg], Any]]=None, unique=False, _explicit=None,
    ):
        if name.startswith('__'):
            raise ValueError('Argument names starting with "__" are reserved for internal arguments.')
        if not name.isidentifier():
            raise ValueError(f'Argument name "{name}" is not a valid identifier.')
        self.name = name
        self.type = type
        if validate and not isinstance(validate, Iterable):
            validate = [validate]
        self.validate = validate
        self.doc = doc
        self.unique = unique
        self.required = required
        self.register = register
        self.children = children
        self.domain = None
        self.value = undefined
        self.prefix = None
        self.post_process_params = None
        if register is not None:
            if type is not str:
                raise ValueError('value register only defined for argument type "str"')
            check_type(f'{name}.register', register, Callable[[str], Callable])
        if post_process:
            if register:
                raise ValueError(f'Argument "{name}" with registry cannot user post_process.')
            if 'type' not in _explicit:
                raise ValueError(f'type should be defined for argument "{name}" with post_process.')
            check_type(f'post_process of argument "{name}"', post_process, Callable)
            argspec = inspect.getfullargspec(post_process)
            post_process_params = len(argspec.args) - (len(argspec.defaults) if argspec.defaults else 0)
            if post_process_params not in (1, 2):
                raise ValueError(f'post_process can only accept 1 or 2 parameters, but {post_process_params} found in "{name}"')
            self.post_process_params = post_process_params
        self._post_process: Optional[Callable[[Any, Arg], Any]] = post_process
        self._explicit = _explicit
        self.default = default
        if 'type' not in _explicit:
            if default is not None:  # else use default type "str"
                default_type = self._type(default)
                if default_type in (str, int, float, bool):
                    self.type = default_type
                else:
                    raise TypeError('non-primitive types should be defined explicitly')
        elif default is not None and not post_process:  # do check only when type is defined explicitly
            check_type(f'default of argument "{name}"', default, self.type)

    def __repr__(self):
        return f'''Arg({self.name})'''
    
    def stringify(self):
        return f'''{self.name}{('*' if self.value != self.default else '')}''', self.value
    
    def schema(self):
        required = ('!' if self.required else '')
        dtype = getattr(self.type, '__name__', str(self.type).split('.')[-1])
        return f'''{required}{self.name}({dtype})''', self.doc or ''

    def set_domain(self, domain: str):
        self.domain = domain
    
    @explicit_checker
    def finalize(self, value=None, final=False, _explicit=None):
        if 'value' not in _explicit:
            if self.required:
                raise AttributeError(f'A required argument "{self.name}" is not provided.')
            value = self.default
        if self._post_process:
            if not final and self.post_process_params > 1:
                # onhold, wait for dependent post_process
                self.value = value if 'value' in _explicit else undefined
                return undefined
            if self.post_process_params == 1:
                assert not final
                value = self._post_process(value)
                if 'value' not in _explicit and self.default is not None:
                    check_type(f'default of argument "{self.name}" after post process', value, self.type)
        if value is not None:
            check_type(self.name, value, self.type)
            if isinstance(self.default, dict) and isinstance(value, dict):
                value.update(self.default)
        for constraint in self.validate:
            decision, errmsg = constraint(value), None
            if isinstance(decision, Iterable):
                decision, errmsg = decision
            if not decision:
                if not errmsg:
                    errmsg = inspect.getsource(constraint)
                errmsg = f'value "{value}" of parameter "{self.name}" does not satisfy the constraint:\n{errmsg}'
                raise ValueError(errmsg)
        self.value = value
        return value

    def post_process(self, args):
        if not self._post_process or self.post_process_params == 1:
            return self.value

        def apply_dependent_post_process(value):
            try:
                return self._post_process(value, args)
            except (TypeError, AttributeError) as e:
                if 'Undefined' in str(e):
                    raise ValueError(
                        f'post_process of argument "{self.name}" cannot depend on other arguments'
                        ' with dependent post_process')
                else:
                    raise ValueError(f'Exception occurs when post process argument "{self.name}" with value: {value}')
            except Exception:
                raise ValueError(f'Exception occurs when post process argument "{self.name}" with value: {value}')
        if self.value is undefined:
            default = apply_dependent_post_process(self.default)
            if default is not None:
                check_type(f'default of argument "{self.name}" after post process', default, self.type)
            value = default
        else:
            value = apply_dependent_post_process(self.value)
        return self.finalize(value, final=True)


class Options:
    def __init__(self):
        self.options = {}
        self._parsed = False
        self._global_constraints = []
        self._set_default_queries = {}
        self._assume_defined_queries = []
        self._mark_required_queries = set()
        self._overwritten_defaults = set()
        self._current_prefix = None
    
    def __contains__(self, name):
        return name in self.options
    
    def __getitem__(self, name):
        return self.options[name]
    
    def items(self):
        return self.options.items()

    def keys(self):
        return self.options.keys()
    
    @contextmanager
    def with_prefix(self, prefix):
        if self._current_prefix is not None:
            raise RuntimeError('Cannot nest with_prefix contexts!')
        self._current_prefix = prefix
        yield
        self._current_prefix = None

    @staticmethod
    def apply_prefix(argument, prefix):
        argument.name = prefix + argument.name
        if argument.children:
            if isinstance(argument.children, list):
                children = argument.children
            else:
                children = [arg for arg_group in argument.children.values() for arg in arg_group]
            for arg in children:
                arg.name = prefix + arg.name
        if argument.register:
            register = argument.register

            def patched_register(value):
                register_fn = register(value)

                def patched_register_fn(options):
                    with options.with_prefix(prefix):
                        register_fn(options)
    
                return patched_register_fn
        
            argument.register = patched_register
        if argument.post_process_params and argument.post_process_params > 1:
            raise ValueError(f'cannot use dependent post_process in argument {argument.name} within prefix {prefix}')

    def register(self, *arguments, domain: Optional[str]=None):
        if self._parsed:
            raise RuntimeError('cannot register new argument to options that are already parsed.')
        for argument in arguments:
            if self._current_prefix:
                self.apply_prefix(argument, self._current_prefix)
            if argument.name in self.options:
                raise ValueError(f'defining duplicated argument "{argument.name}".')
            self.options[argument.name] = argument
            if domain:
                self.options[argument.name].set_domain(domain)
    
    def set_default(self, name, value, strict=True):
        """param "strict": whether to give an error or warning if
        setting a default value to a name that is not defined"""
        if self._current_prefix:
            name = self._current_prefix + name
            strict = False
        if name in self._set_default_queries:
            raise ValueError('Cannot set defaults to the same argument twice: '
                             'the final value will not be determined due to random parse order.')
        self._set_default_queries[name] = (value, strict)
        
    def _set_default(self, name, value, strict):
        if name not in self.options:
            msg = f'Setting defaults to an argument "{name}" that is not defined.'
            if strict:
                raise ValueError(msg)
            # print(f'WARNING: {msg}')
            return
        argument = self.options[name]
        if argument.register is not None or argument.children:
            raise ValueError(f'Cannot set defaults to arguments with register hooks or children: "{name}"')
        if name in self._overwritten_defaults:
            raise ValueError(f'Cannot set defaults to the same argument twice: '
                             f'the final value will not be determined due to random parse order.')
        # verify whether new default value statisfy the original constraints
        argument.finalize(value)
        if argument.required:
            argument.required = False
        argument.default = value
        self._overwritten_defaults.add(name)
    
    def add_global_constraint(self, constraint: Callable[[Arg], Union[bool, Tuple]]):
        check_type('constraint', constraint, Callable)
        if self._current_prefix:
            prefix = self._current_prefix

            def patched_constraint(args):
                return constraint(args.strip_prefix(prefix))
        else:
            patched_constraint = constraint
        self._global_constraints.append(patched_constraint)
    
    def assume_defined(self, name, by, type=None,):
        self._assume_defined_queries.append(dict(name=name, type=type, by=by))

    def mark_required(self, name):
        self._mark_required_queries.add(name)

    def _mark_required(self, name):
        if name not in self.options:
            raise ValueError(f'Marking required for an argument "{name}" that is not defined.')
        argument = self.options[name]
        argument.required = True
        argument.default = None

    def tree(self, args=None):
        if not self._parsed and not args:
            return '<unparsed Options object>'
        if args:
            options = deepcopy(self).restore(args)
        else:
            options = self
        sorted_options = sorted(options.options.values(), 
            key=lambda argument: len(argument.domain) if argument.domain else 0)

        class CleanOrderedDict(OrderedDict):
            def __repr__(self):
                return dict.__repr__(self)
        
        def schema_factory(fn):
            arguments = CleanOrderedDict()
            for argument in sorted_options:
                domain = argument.domain
                subset = arguments
                if domain:
                    for domain in domain.split('/'):
                        domain = f'({domain})'
                        if domain not in subset:
                            subset[domain] = CleanOrderedDict()
                        subset = subset[domain]
                key, val = fn(argument)
                subset[key] = val
            return arguments
        
        arguments = schema_factory(lambda argument: argument.stringify())
        arguments = unsorted_pformat(arguments)
        return arguments

    def parse(self, config_file: str) -> List[Arg]:
        """parse args with __parent__"""
        configs = self.load_configs_from_file(config_file)
        args_list = []
        unique_arguments = {}
        for config in configs:
            # dynamic arguments are loaded for each config, the original options object is kept untouched
            options = deepcopy(self)
            args = options.parse_dict(config)
            args_list.append(args)
            for name, argument in options.items():
                if argument.unique:
                    if name not in unique_arguments:
                        unique_arguments[name] = set()
                    value = getattr(args, name)
                    if value in unique_arguments[name]:
                        raise ValueError(f'values for "{name}" must be unique. Duplications found: {value}"')
                    unique_arguments[name].add(value)

        return args_list

    @staticmethod
    def pai_config():
        return {
            '__gpu__': 'n_gpus',
            '__cpu__': 'n_cpus',
            '__memory__': 'memory',
            '__worker__': 'n_workers',
            '__pytorch__': 'pt_version',
            '__v100__': 'v100',
        }

    @staticmethod
    def load_configs_from_file(config_file: str) -> List[dict]:
        """load configs as dict from config files with their parents"""
        init_configs = Options.load_hjson(config_file)
        # save pai_config from the surface layer
        pai_config = {}
        for key, val in Options.pai_config().items():
            if key in init_configs:
                pai_config[val] = init_configs.pop(key)
        if pai_config:
            with open('.pai_config', 'a') as f:
                for key, val in pai_config.items():
                    f.write(f'{key}={val}\n')
        # recursively load parents
        root = os.path.dirname(config_file)
        if isinstance(init_configs, dict):
            init_configs = [init_configs]
        configs = []
        for init_config in init_configs:
            configs.extend(Options.load_parents(init_config, root))
        for i, config in enumerate(configs):
            config['__origin__'] = config_file + f'.{i}'
        return configs

    @staticmethod
    def parse_starter_config(arguments: str) -> dict:
        """"arguments" is the return value of tree()"""
        tree = eval(arguments)
        arguments = OrderedDict()

        def _parse_tree(tree):
            """choose arguments that're different from origins (endswith *)"""
            for key, val in tree.items():
                if re.match(r'\(.+\)', key):
                    _parse_tree(val)
                elif key.endswith('*'):
                    arguments[key.rstrip('*')] = val

        _parse_tree(tree)

        if 'data_dir' in arguments:
            meta_path = os.path.join(arguments['data_dir'], 'meta.hjson')
            from xdpx.utils import io
            if io.exists(meta_path):
                meta = Options.load_hjson(meta_path)
                for key in meta:
                    if key in arguments:
                        del arguments[key]
                arguments['__parent__'] = '${data_dir}/meta'
                arguments.move_to_end('__parent__', last=False)
        arguments.pop('distributed_world_size', None)
        arguments.pop('device_id', None)
        return arguments

    from xdpx.utils import get_commit_hash, is_apex_available
    _builtin_def = {
        '__commit_hash__': get_commit_hash,
        '__date__': lambda: datetime.now().strftime("%y%m%d"),
        '__gpus__': lambda: torch.cuda.device_count() if 'VISIBLE_DEVICE_LIST' not in os.environ else 1,
        '__workers__': lambda: int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1,
        '__apex__': is_apex_available,
        '__dc7__': lambda: torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7,
        'test_root': lambda: open('.test_meta').read().rstrip('\n').rstrip('/'),
    }

    @staticmethod
    def apply_var(value, config: dict):
        if isinstance(value, dict):
            return {Options.apply_var(key, config): Options.apply_var(val, config) for key, val in value.items()}
        if not isinstance(value, str):
            return value
    
        def replace(m):
            name = m.group(1)
            definitions = config.get('__def__', {})
            if name in definitions:
                value = definitions[name]
            elif name in config:
                value = config[name]
            elif name in Options._builtin_def:
                value = Options._builtin_def[name]()
            else:
                raise KeyError(f'config variable "${{{name}}}" not defined in "{origin_value}".')
            if isinstance(value, str):
                value = value.rstrip('/')
            return value
        
        origin_value = value
        for _ in range(128):
            m = re.search(r'\${(.+?)}', value)
            if not m:
                break
            start, end = m.span()
            if start == 0 and end == len(value):
                return replace(m)
            value = value[:start] + str(replace(m)) + value[end:]
        else:
            raise ValueError(f'Maximum recursion reached. There are probably cycle definitions while parsing "{origin_value}"')
        return value
        
    @staticmethod
    def merge_config(*configs):
        """higher-level config should come first"""
        merged = copy(configs[0])
        ignored = merged.get('__ignore__', [])
        if not isinstance(ignored, list):
            ignored = [ignored]
        ignored = set(ignored)
        for config in configs[1:]:
            for key, val in config.items():
                if key in ignored:
                    continue
                if key == '__ignore__':
                    if not isinstance(val, list):
                        val = [val]
                    ignored.update(val)
                if key not in merged:
                    merged[key] = val
                elif key == '__def__':
                    definitions = merged[key]
                    for key, val in val.items():
                        if key not in definitions:
                            definitions[key] = val
                elif isinstance(merged[key], dict) and isinstance(val, dict):
                    merged[key] = {**val, **merged[key]}
        if ignored:
            merged['__ignore__'] = list(ignored)
        return merged
    
    @staticmethod
    def load_parents(init_config: dict, root: str) -> List[dict]:
        """load parents of init_config"""
        parents = init_config.pop('__parent__', [])

        def _load_parents(parents, root, init_config):
            """load parents and merge their configs in a top-down manner"""
            if isinstance(parents, str):
                parents = [parents]
            assert isinstance(parents, list)
            configs = [init_config] 
            for unparsed_parent in parents[::-1]:
                new_configs = []
                for orig_config in configs:
                    prefix = ''
                    if isinstance(unparsed_parent, str):
                        m = re.search(r'>>', unparsed_parent)
                        if m:
                            prefix = unparsed_parent[m.end():].strip()
                            unparsed_parent = unparsed_parent[:m.start()].strip()
                        # must use a name different from the one in the outer loop,
                        # otherwise the latter will be overwritten
                        parent = Options.apply_var(unparsed_parent, orig_config)
                        parent = Options.parse_relative_path(parent, root)
                        parent_configs = Options.load_hjson(parent)
                        new_root = os.path.dirname(parent)
                    else:
                        parent_configs = unparsed_parent
                        new_root = root
                    if isinstance(parent_configs, dict):
                        parent_configs = [parent_configs]
                    for parent_config in parent_configs:
                        # recursively load grandparents
                        if not isinstance(parent_config, dict):
                            raise TypeError('Child of a config-level list should be a dict.')
                        parent_config = deepcopy(parent_config)
                        config = deepcopy(orig_config)
                        grandparents = parent_config.pop('__parent__', [])
                        if prefix:
                            assert not grandparents, 'prefix cannot take effect on grandparents'
                            parent_config = {prefix + key: val for key, val in parent_config.items()}
                        config = Options.merge_config(config, parent_config)
                        full_parent_configs = _load_parents(
                            grandparents,
                            new_root,
                            config,
                        )
                        new_configs.extend(full_parent_configs)
                configs = new_configs
            return configs

        configs = _load_parents(parents, root, init_config)
        results = []
        for config in configs:
            config = {key: Options.apply_var(value, config) if key != '__def__' else value for key, value in config.items()}
            config.pop('__def__', None)
            for key in config.pop('__ignore__', []):
                if key in config:
                    del config[key]
            results.append(config)
        return results

    def restore(self, args: Arg):
        """restore params from parsed args"""
        config = vars(args)
        self.parse_dict(config)
        # restore value for non-idempotent arguments
        for name, argument in self.options.items():
            argument.value = getattr(args, name)
        return self
    
    @staticmethod
    def parse_tree(tree: dict):
        """parse args from saved tree structure"""
        args = Arg()
        def _parse_tree(tree):
            for key, val in tree.items():
                if re.match(r'\(.+\)', key):
                    _parse_tree(val)
                else:
                    key = key.rstrip('*')
                    setattr(args, key, val)
        _parse_tree(tree)
        Options.add_internal_args(args)
        return args

    def parse_dict(self, config: dict):
        """parse args from a dict object, will add dynamic arguments while parsing"""
        # recursively load dynamic schema
        diff_option_keys = set(self.options.keys())
        while diff_option_keys:
            prev_option_keys = set(self.options.keys())
            for name in self.options.keys():
                if name in self._set_default_queries:
                    value, strict = self._set_default_queries.pop(name)
                    self._set_default(name, value, strict)
                if name in self._mark_required_queries:
                    self._mark_required(name)
            for name in diff_option_keys:
                argument = self.options[name]
                if argument.register is not None or argument.children:
                    if name in config:
                        value = argument.finalize(config[name])
                    else:
                        value = argument.finalize()
                    if argument.register is not None:
                        argument.register(value)(WrappedOptions(self, argument.domain))
                    if value is undefined:
                        # use the original value as the indicator for children with dependent post-processing
                        value = config[name] if name in config else argument.default
                    if isinstance(argument.children, list):
                        # register children if value is True
                        if argument.children and value:
                            domain = argument.domain + '/' + argument.name if argument.domain else argument.name
                            self.register(*argument.children, domain=domain)
                    else:
                        for criterion, children in argument.children.items():
                            if criterion(value):  # register corresponding children if the specified creterion is met
                                self.register(*children, domain=argument.domain)
            diff_option_keys = self.options.keys() - prev_option_keys

        unknown = config.keys() - self.options.keys()
        # skip internal arguments
        unknown = [name for name in unknown if not name.startswith('__')]
        if unknown:
            raise AttributeError('unknown parameter names: ' + ' '.join(f'"{name}"' for name in unknown))
        
        for name, (value, strict) in self._set_default_queries.items():
            self._set_default(name, value, strict)
        
        for item in self._assume_defined_queries:
            name = item['name']
            _type = item['type']
            by = item['by']
            if name not in self.options:
                raise RuntimeError(f'Argument {name} is assumed defined by "{by}"')
            if _type is not None:
                if self.options[name].type != _type:
                    raise RuntimeError(f'Argument is assumed to be type {_type} by "{by}", but type {self.options[name].type} found.')

        args = Arg()
        for name, argument in self.options.items():
            if name in config:
                value = argument.finalize(config[name])
            else:
                value = argument.finalize()
            setattr(args, name, value)

        if '__origin__' in config:
            args.__origin__ = config['__origin__']
        
        for name, argument in self.options.items():
            value = argument.post_process(args)
            setattr(args, name, value)
        
        for constraint in self._global_constraints:
            decision, errmsg = constraint(args), None
            if isinstance(decision, Iterable):
                decision, errmsg = decision
            if not decision:
                if not errmsg:
                    errmsg = inspect.getsource(constraint)
                raise ValueError(f'Global constraint not satisfied: {errmsg}\n')
        self._parsed = True
        args = Options.add_internal_args(args)
        return args

    @staticmethod
    def add_internal_args(args):
        return args

    @staticmethod
    def parse_relative_path(path, root):
        if not path.startswith('.'):
            return path

        def _parse_relative_path(path, root):
            if path.startswith('./'):
                return _parse_relative_path(path[2:], root)
            if path.startswith('../'):
                return _parse_relative_path(path[3:], os.path.dirname(root))
            return os.path.join(root, path)

        return _parse_relative_path(path, root)

    @staticmethod
    def load_hjson(file: str, root='') -> dict:
        from .utils import io

        file = Options.parse_relative_path(file, root)

        if not file.endswith('.hjson') and not file.endswith('.json'):
            file += '.hjson'

        def unique_dict(ordered_pairs):
            d = {}
            for k, v in ordered_pairs:
                if k in d:
                    raise KeyError(f'duplicate key in config: "{k}"')
                else:
                    d[k] = v
            return d

        with io.open(file) as f:
            config = hjson.load(f, object_pairs_hook=unique_dict)
            return config
    
    @staticmethod
    def save_hjson(config: Union[Arg, Dict[str, Any]], path: str):
        from .utils import io
        if isinstance(config, Arg):
            config = vars(config)
        with io.open(path, 'w') as f:
            hjson.dump(dict((key, val) for key, val in config.items() if not key.startswith('__')),
                       f, sort_keys=True)
    

class WrappedOptions:
    def __init__(self, options, domain):
        self.options = options
        self.domain = domain
    
    def register(self, *arguments, domain=None):
        current_domain = self.domain
        if domain:
            if current_domain:
                current_domain += '/' + domain
            else:
                current_domain = domain
        self.options.register(*arguments, domain=current_domain)

    def __getattr__(self, name):
        if hasattr(self.options, name):
            return getattr(self.options, name)
        return super().__getattr__(name)


class Undefined: pass
undefined = Undefined()


def unsorted_pformat(obj):
    import pprint
    # monkey patch
    pprint.sorted = lambda x, key=None: x
    return pprint.pformat(obj, width=120)


def parse_relative(value, args: Arg, reverse=True):
    """parse something like: @{batch_size:64}/1200"""
    if not isinstance(value, str):
        return value
    m = re.match(r'@{([^:]+):([ 0-9.+-]+)}/([ 0-9.+-]+)', value)
    if m is None:
        return value
    ref_name = m.group(1).strip()
    ref_val = getattr(args, ref_name)
    ref_default = float(m.group(2))
    default = float(m.group(3))
    try:
        int(m.group(2))
        int(m.group(3))
        int(str(ref_val))
        transform = int
    except ValueError:
        transform = lambda x: x
    if reverse:
        return transform(ref_default / ref_val * default)
    else:
        return transform(ref_val / ref_default * default)
