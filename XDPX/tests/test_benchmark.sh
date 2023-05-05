set -e

x-prepro tests/benchmark/finetune/cls/prepro
x-train tests/benchmark/finetune/cls/train
x-prepro tests/benchmark/finetune/match/prepro
x-train tests/benchmark/finetune/match/train

x-prepro tests/benchmark/adv/prepro
x-train tests/benchmark/adv/train
