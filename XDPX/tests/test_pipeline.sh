# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
function message {
    if [[ $3 -ne 0 ]] ; then
        echo "line $1: \"$2\" failed with exit code $3."
    else
        echo
        echo "All pipeline tests run without error."
    fi
}
trap 'message $LINENO "$last_command" $?' EXIT

outdir=/tmp/outputs/debug/
[ -e $outdir ] && rm -r $outdir
echo $outdir > .test_meta

# test for pretraining bert with pairwise labels
x-prepro tests/sample_tasks/pretrain/click/prepro.hjson
x-train tests/sample_tasks/pretrain/click/train.hjson
x-pred tests/sample_tasks/pretrain/click/predict.hjson
x-script export_tf tests/sample_tasks/pretrain/click/export_tf.hjson
x-script export_pt tests/sample_tasks/pretrain/click/export_back.hjson

# test for MLM pretraining (with and without whole word masking)
x-prepro tests/sample_tasks/pretrain/wiki/prepro.hjson
x-script prune_heads tests/sample_tasks/pretrain/wiki/prune.hjson
x-train tests/sample_tasks/pretrain/wiki/train.hjson
x-train tests/sample_tasks/pretrain/wiki/train_pruned.hjson
x-train tests/sample_tasks/pretrain/wiki/train_albert.hjson

x-prepro tests/sample_tasks/pretrain/wiki_wwm/prepro.hjson
x-train tests/sample_tasks/pretrain/wiki_wwm/train.hjson
x-train tests/sample_tasks/pretrain/wiki_wwm/train_rezero.hjson

x-prepro tests/sample_tasks/pretrain/structbert/prepro.hjson
x-train tests/sample_tasks/pretrain/structbert/train.hjson

# test for RMR chatlog pretraining
x-prepro tests/sample_tasks/pretrain/chatlog/prepro.hjson
x-train tests/sample_tasks/pretrain/chatlog/train.hjson
x-pred tests/sample_tasks/pretrain/chatlog/pred.hjson

# test for finetuning Bert for text match tasks
x-prepro tests/sample_tasks/finetune/fushi/prepro.hjson
x-train tests/sample_tasks/finetune/fushi/train.hjson
x-eval tests/sample_tasks/finetune/fushi/eval.hjson
x-pred tests/sample_tasks/finetune/fushi/predict.hjson
x-script export_torchscript tests/sample_tasks/finetune/fushi/export.hjson

# test for match-cnn distillation
x-prepro tests/sample_tasks/distill/match_cnn/prepro.hjson
x-train tests/sample_tasks/distill/match_cnn/train.hjson

# test for finetuning Bert with siamese loss
x-prepro tests/sample_tasks/finetune/fushi_siamese/prepro.hjson
x-train tests/sample_tasks/finetune/fushi_siamese/train.hjson

# test for finetuning Bert for classification tasks
x-prepro tests/sample_tasks/finetune/cainiao/prepro.hjson
x-train tests/sample_tasks/finetune/cainiao/train.hjson
x-eval tests/sample_tasks/finetune/cainiao/eval.hjson
x-pred tests/sample_tasks/finetune/cainiao/predict.hjson
#x-script export_tf tests/sample_tasks/finetune/cainiao/export.hjson
x-train tests/sample_tasks/finetune/cainiao/train_adv.hjson

# test for bert_mix finetuning with BCE loss
x-prepro tests/sample_tasks/finetune/cainiao_mix/prepro.hjson
x-train tests/sample_tasks/finetune/cainiao_mix/train.hjson
x-eval tests/sample_tasks/finetune/cainiao_mix/eval.hjson
x-pred tests/sample_tasks/finetune/cainiao_mix/pred.hjson
x-script bce_threshold tests/sample_tasks/finetune/cainiao_mix/threshold.hjson
x-pred tests/sample_tasks/finetune/cainiao_mix/distill.hjson

# test for cls-cnn distillation
x-prepro tests/sample_tasks/distill/cls_cnn/prepro.hjson
x-train tests/sample_tasks/distill/cls_cnn/train.hjson
x-eval tests/sample_tasks/distill/cls_cnn/eval.hjson
x-script export_tf tests/sample_tasks/distill/cls_cnn/export.hjson

# test for RE2 training
x-prepro tests/sample_tasks/classic/fushi/prepro.hjson
x-train tests/sample_tasks/classic/fushi/train.hjson
x-eval tests/sample_tasks/classic/fushi/eval.hjson
x-pred tests/sample_tasks/classic/fushi/predict.hjson
x-script export_tf tests/sample_tasks/classic/fushi/export_tf.hjson
x-script export_torchscript tests/sample_tasks/classic/fushi/export.hjson
x-train tests/sample_tasks/classic/fushi/trainA.hjson
x-train tests/sample_tasks/classic/fushi/trainB.hjson

# test for AutoML
# TODO add local cache to automl
#x-prepro tests/sample_tasks/classic/fushi/prepro.hjson
#x-tune tests/sample_tasks/automl/random/tune.hjson
#x-tune tests/sample_tasks/automl/grid/tune.hjson

# test for tinybert distillation
x-prepro tests/sample_tasks/distill/tinybert_gd/prepro.hjson
x-train tests/sample_tasks/distill/tinybert_gd/train.hjson

# test for minilm distillation
x-prepro tests/sample_tasks/distill/minilm/prepro.hjson
x-train tests/sample_tasks/distill/minilm/train.hjson

# test for LOPKM initialzation and distillation
x-prepro tests/sample_tasks/pretrain/wiki/prepro.hjson
x-script pkm_init tests/sample_tasks/distill/lopkm/init.hjson
x-train tests/sample_tasks/distill/lopkm/train.hjson
# TODO: fix train/eval differences in lopkm distill
#x-eval tests/sample_tasks/distill/lopkm/eval.hjson
x-train tests/sample_tasks/distill/lopkm/minilm.hjson
x-train tests/sample_tasks/distill/lopkm/train_inserted.hjson
x-eval tests/sample_tasks/distill/lopkm/eval_inserted.hjson
