#!/usr/bin/env bash
cd ..

sigma=$1
steps=$2
reverse_seed=$3

python eval_certified_densepure.py \
--exp exp/cifar10 \
--config cifar10.yml \
--domain cifar10 \
--seed 0 \
--diffusion_type ddpm \
--lp_norm L2 \
--outfile results \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 0 20 9980) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_t_steps \
--num_t_steps $steps \
--save_predictions \
--predictions_path exp/\
--reverse_seed $reverse_seed

