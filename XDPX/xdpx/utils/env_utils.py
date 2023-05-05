import importlib
import subprocess
import multiprocessing
import shlex
from functools import partial

from . import register

dependencies = {}
register = partial(register, registry=dependencies)


def _check_installed(name, output):
    try:
        importlib.import_module(name)
        print(f'{name} already installed. Skip installation.')
        flag = True
    except:
        flag = False
    output.append(flag)


def check_installed(name):
    mp = multiprocessing.get_context('spawn')
    with multiprocessing.Manager() as manager:
        flag_holder = manager.list()
        process = mp.Process(
            target=_check_installed,
            args=(name, flag_holder),
            daemon=False,
        )
        process.start()
        process.join()
        return flag_holder.pop()


@register('tensorflow')
def install_tensorflow(tf_version=None):
    if check_installed('tensorflow'):
        return
    if not tf_version:
        import torch
        from xdpx.utils.versions import torch_lt_120
        tf_version = '1.10' if torch_lt_120() else '1.14'
    channel = 'https://pypi.tuna.tsinghua.edu.cn/simple'
    command = f'sudo /opt/conda/envs/python3.6/bin/pip install tensorflow=={tf_version} -i {channel}'
    subprocess.run(shlex.split(command))

    command = 'sudo ldd /opt/conda/envs/python3.6/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so'
    subprocess.run(shlex.split(command))


@register('faiss')
def install_faiss():
    if check_installed('faiss'):
        return
    import torch
    if torch.cuda.is_available():
        cuda_version = '.'.join(torch.version.cuda.split('.')[:2])
        channel = 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/'
        command = f'sudo /opt/conda/bin/conda install -y -n python3.6 faiss-gpu cudatoolkit={cuda_version} -c {channel}'
        subprocess.run(shlex.split(command))

        command = 'sudo ldd /opt/conda/envs/python3.6/lib/python3.6/site-packages/faiss/_swigfaiss_avx2.so'
        subprocess.run(shlex.split(command))
    else:
        install_faiss_cpu()


@register('faiss-cpu')
def install_faiss_cpu():
    if check_installed('faiss'):
        return
    channel = 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/'
    command = f'sudo /opt/conda/bin/conda install -y -n python3.6 faiss-cpu -c {channel}'
    subprocess.run(shlex.split(command))
