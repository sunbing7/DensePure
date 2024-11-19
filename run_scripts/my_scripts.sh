python eval_certified_densepure.py \
--exp /root/autodl-tmp/sunbing/workspace/uap/my_result/densepure/results/exp/cifar10 \
--config cifar10.yml \
-i cifar10-densepure-sample_num_10000-noise_1.0-2-3 \
--domain cifar10 \
--seed 0 \
--diffusion_type ddpm \
--lp_norm L2 \
--outfile /root/autodl-tmp/sunbing/workspace/uap/my_result/densepure/results/cifar10-densepure-sample_num_10-noise_1.0-2-steps-0 \
--sigma 1 \
--N 10000 \
--N0 10 \
--certified_batch 100 \
--sample_id 0 20 9980 \
--use_id \
--certify_mode purify \
--advanced_classifier cifar10-wideresnet \
--use_t_steps \
--num_t_steps 2 \
--save_predictions \
--predictions_path /root/autodl-tmp/sunbing/workspace/uap/my_result/densepure/results/exp/cifar10/1- \
--reverse_seed 3

python merge_results.py \
--sample_id_list 0 20 9980 \
--sample_num 3 \
--majority_vote_num 1 \
--N 10000 \
--N0 10 \
--sigma 1.0 \
--classes_num 10 \
--datasets cifar10 \
--steps 2



python eval_certified_densepure.py \
--exp results/exp/cifar10 \
--config cifar10.yml \
--domain cifar10 \
--seed 0 \
--diffusion_type ddpm \
--lp_norm L2 \
--outfile results/ \
--sigma 1.0 \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--certify_mode purify \
--advanced_classifier cifar10-wideresnet \
--use_t_steps \
--num_t_steps 10 \
--save_predictions \
--reverse_seed 0

python eval_certified_densepure.py \
--exp results/exp/cifar10 \
--config cifar10.yml \
-i cifar10-densepure-sample_num_100000-noise_1.0-2-1 \
--domain cifar10 \
--seed 0 \
--diffusion_type ddpm \
--lp_norm L2 \
--outfile results/cifar10-densepure-sample_num_100000-noise_1.0-2-steps-1 \
--sigma 1.0 \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id 0 20 9980 \
--use_id \
--certify_mode purify \
--advanced_classifier cifar10-wideresnet \
--use_t_steps \
--num_t_steps 2 \
--save_predictions \
--predictions_path results/exp/cifar10/1.0- \
--reverse_seed 1

python eval_certified_densepure.py \
--exp results/exp/cifar10 \
--config cifar10.yml \
-i cifar10-densepure-sample_num_100000-noise_1.0-2-2 \
--domain cifar10 \
--seed 0 \
--diffusion_type ddpm \
--lp_norm L2 \
--outfile results/cifar10-densepure-sample_num_100000-noise_1.0-2-steps-2 \
--sigma 1.0 \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id 0 20 9980 \
--use_id \
--certify_mode purify \
--advanced_classifier cifar10-wideresnet \
--use_t_steps \
--num_t_steps 2 \
--save_predictions \
--predictions_path results/exp/cifar10/1.0- \
--reverse_seed 2


python merge_results.py \
--sample_id_list 0 20 9980 \
--sample_num 3 \
--majority_vote_num 3 \
--N 100000 \
--N0 100 \
--sigma 1.0 \
--classes_num 10 \
--datasets cifar10 \
--steps 2

python merge_results.py --sample_num 100 --max 10000 --skip 100 --majority_vote_num 10 --N 100 --N0 10 --sigma 0.25 --classes_num 10 --datasets cifar10 --steps 10
rm cifar10-densepure*
rm exp/0.25*
rm -r exp/cifar10-densepure*