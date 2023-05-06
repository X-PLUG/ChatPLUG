import sys
from setuptools import setup, find_packages
from xdpx import __version__ as xdpx_version

if sys.version_info < (3, 6):
    sys.exit('Python >= 3.6 is required')

with open('ReadMe.md',encoding="utf-8") as f:
    readme = f.read()

setup(
    name='xdpx',
    version=xdpx_version,
    description='DeepQA Platform X',
    url='https://gitlab.alibaba-inc.com/deepqa/xdpx',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    long_description=readme,
    long_description_content_type='text/markdown',
    setup_requires=[
        'setuptools>=18.0',
    ],
    install_requires=[
        'packaging>=20.9',
        'psutil',
        'tqdm>=4.48',
        'numpy>=1.17',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'typeguard==2.13.3',
        'hjson',
        'nvidia-ml-py3',
        'oss2>=2.7.0',
        'coverage',
        'tabulate',
        'filelock',
        'importlib_metadata',
        'transformers==4.13.0',
        #'torch>=1.8.1',
        'google',
        #'torchaudio',
        'lxml',
        'sentencepiece',
        'protobuf',
        'flask',
        'waitress',
        'jieba',
        'requests_html',
        'icecream',
        'easydict',
        # 'onnxruntime-gpu',
        'alibabacloud_chatbot20220408==1.0.4',
        'lunar_python',
        'rouge',
        'coloredlogs',
        'psutil',
        'py-cpuinfo',
        'py3nvml',
        'dacite',
        'elasticsearch==7.16',
        'sympy',
        # 'onnx',
        # 'polygraphy'
        # 'faiss-gpu'
        'pytorch_pretrained_bert',
        'arrow==0.14.0',
        'rank_bm25',
        'cacheout',
        'fasttext',
        'deepspeed'

    ],
    packages=find_packages(exclude=['user*']),
    package_data={'': ['*.txt']},
    test_suite='tests',
    entry_points={
        'console_scripts': [
            'x-prepro = xdpx.run:prepro_entry',
            'x-train = xdpx.run:train_entry',
            'x-viz = xdpx.run:viz_entry',
            'x-pred = xdpx.run:pred_entry',
            'x-eval = xdpx.run:eval_entry',
            'x-tune = xdpx.run:tune_entry',
            'x-script = xdpx.run:script_entry',
            'x-io = xdpx.run:io_entry',
        ],
    },
    zip_safe=False,
)
