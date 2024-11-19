#!/usr/bin/env bash

sigma=$1
steps=$2
majority_vote_num=$3

python merge_results.py \
--sample_id_list $(seq -s ' ' 0 20 9980) \
--sample_num 500 \
--majority_vote_num $majority_vote_num \
--N 100000 \
--N0 100 \
--sigma $sigma \
--classes_num 10 \
--datasets cifar10 \
--steps $steps


python merge_result_ours.py --uap_target 0 --sample_num 100 --max 10000 --skip 100 --majority_vote_num 40 --N 100 --N0 10 --sigma 0.25 --classes_num 10 --datasets cifar10 --steps 10