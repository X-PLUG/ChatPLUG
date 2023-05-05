

# PLUG-dialog 

## QuickStart

### Run on DSW
```bash
cd XDPX
pip install -e .

# update the following files
user/modules/oss_credentials.py
```

### Run on PAI

```bash
# add java&odps

# run
PYTHONPATH='./' bash pai.sh script
```

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH='./' x-script seq2seq_chat oss://141457d2/plug-dialog/chat/plug_base
CUDA_VISIBLE_DEVICES=0 PYTHONPATH='./' x-script seq2seq_chat ../chat_data/data_processed/v0.9/1e-05/

CUDA_VISIBLE_DEVICES=0 PYTHONPATH='./' x-script fid_t5chat examples/chat/plug_chat_v0.1.hjson

```

## Finetune

oss://141457d2/plug-dialog/chat/data_processed/v0.9.3

```bash
PYTHONPATH='./' x-prepro examples/chat/plug_prepro_v0.1.hjson
CUDA_VISIBLE_DEVICES=1 PYTHONPATH='./' 

bash pai.sh 
CUDA_VISIBLE_DEVICES=0 PYTHONPATH='./' x-train examples/chat/plug_train_v0.1.hjson

PYTHONPATH='./' x-train examples/chat/plug_train_v0.2.dsw.hjson
```


## SelfChat

```config
#test_self_chat
```