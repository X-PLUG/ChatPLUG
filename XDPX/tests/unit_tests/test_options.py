import unittest
from typing import List

from xdpx.options import Options, Argument


class Adam:
    @staticmethod
    def register(options: Options):
        options.register(
            Argument('betas', type=List[float], validate=(
                lambda x: len(x) == 2,
                lambda x: all(i > 0 for i in x),
            ), default=[0.9, 0.999], doc='beta for adam'),
            Argument('amsgrad', default=False),
        domain='adam')
        options.set_default('learning_rate', 1e-3)
    
class SGD:
    @staticmethod
    def register(options):
        pass

registry = {'adam': Adam, 'sgd': SGD}


class TestOptions(unittest.TestCase):
    def setUp(self):
        options = Options()
        options.register(
            Argument('learning_rate', type=float, doc='learning rate', validate=lambda x: x>0),
            Argument(
                'optimizer', default='adam', 
                validate=lambda value: value.lower() in ['adam', 'sgd', 'adagrad'],
                register=lambda value: registry[value].register,
            ),
            Argument('warmup_steps', default=0),
        domain='trainer/optimizer')
        options.register(
            Argument('max_epochs', type=int),
            Argument('batch_size', default=128),
        domain='trainer')
        options.register(
            Argument('data_dir', required=True),
            Argument('save_dir', default='output', unique=True),
        )
        self.options = options

    def test_default(self):
        config_file = 'tests/options/default.hjson'
        args = self.options.parse(config_file)[0]
        # in config file
        self.assertEqual(args.learning_rate, 5e-4)
        self.assertEqual(args.betas, [0.9, 0.99])
        self.assertEqual(args.max_epochs, 10)
        # not in config file
        self.assertEqual(args.amsgrad, False)
        self.assertEqual(args.batch_size, 128)

    def test_user(self):
        config_file = 'tests/options/user.hjson'
        args = self.options.parse(config_file)[0]
        self.assertEqual(args.amsgrad, True)
        self.assertEqual(args.learning_rate, 4e-4)
        self.assertEqual(args.max_epochs, 10)
        self.assertEqual(args.betas, [0.9, 0.999])
    
    def test_batch_config(self):
        config_file = 'tests/options/main.hjson'
        arg_group = self.options.parse(config_file)
        self.assertEqual(len(arg_group), 3)
        for args, batch_size in zip(arg_group, [32, 64, 128]):
            self.assertEqual(args.batch_size, batch_size)
            self.assertEqual(args.amsgrad, True)
            self.assertEqual(args.learning_rate, 5e-4)
            self.assertEqual(args.data_dir, '/path/to/root/data')
    
    def test_variable_omission(self):
        config_file = 'tests/options/main_omit.hjson'
        with self.assertRaises(KeyError):
            self.options.parse(config_file)
        
    def test_duplicate(self):
        config_file = 'tests/options/main_dup.hjson'
        with self.assertRaises(ValueError):
            self.options.parse(config_file)
    
    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            config = {
                'data_dir': 'data',
                'optimizer': 'adam',
                'learning_rate': 5e-4,
                'betas': 0.99,  # this type is wrong
            }
            self.options.parse_dict(config)

    def test_dynamic_loading(self):
        "use unknown params"
        with self.assertRaises(AttributeError):
            config = {
                'data_dir': 'data',
                'optimizer': 'sgd',
                'learning_rate': 5e-4,
                'betas': [0.9, 0.99],  # this param is not in "SGD"
            }
            self.options.parse_dict(config)

    def test_invalid(self):
        "not match the requirements of 'validate'"
        with self.assertRaises(ValueError):
            config = {
                'data_dir': 'data',
                'optimizer': 'adam',
                'learning_rate': 5e-4,
                'betas': [0, 0.99],  # the value is not in the correct range
            }
            self.options.parse_dict(config)
    
    def test_omit(self):
        "omit the required arguments"
        with self.assertRaises(AttributeError):
            config = {
                'optimizer': 'adam',
                'learning_rate': 5e-4,
            }
            self.options.parse_dict(config)
    
    def test_call_unknown(self):
        "call unknown arguments in other modules which is not dynamic loaded"
        config = {
            'data_dir': 'data',
            'optimizer': 'sgd',
            'learning_rate': 5e-4,
        }
        args = self.options.parse_dict(config)
        with self.assertRaises(AttributeError):
            args.betas
    
    def test_duplicated_config(self):
        options = Options()
        options.register(
            Argument('max_epochs', type=int),
            Argument('batch_size', type=int, default=128),
        domain='trainer')
        with self.assertRaises(ValueError):
            options.register(
                Argument('batch_size', type=int, default=128)
            )
        
    def test_wrong_default_type(self):
        options = Options()
        with self.assertRaises(TypeError):
            options.register(
                Argument('max_epochs', type=int),
                Argument('batch_size', type=int, default=128.0),
            domain='trainer')
        

if __name__ == "__main__":
    unittest.main()