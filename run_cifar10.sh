
SIGMA=0.25
#SIGMA=0.04
STEPS=10
N=100
N0=10

for TARGET_CLASS in {0,1,2,3,5,4,6,7,8,9}
do
    for REVSEED in {0..39}
    do
      echo "Target class:" $TARGET_CLASS ", maj_vote:" $REVSEED
      python eval_certified_densepure.py --uap_test 1 --uap_target $TARGET_CLASS --max -1 --skip 100 --exp results/exp/cifar10 --config cifar10.yml --domain cifar10 --seed 0 --diffusion_type ddpm --lp_norm L2 --outfile results/ --sigma $SIGMA --N $N --N0 $N0 --certified_batch 100 --certify_mode purify --advanced_classifier cifar10-wideresnet --use_t_steps --num_t_steps $STEPS --save_predictions --predictions_path results/exp/ --reverse_seed=$REVSEED
    done
done
#REVSEED=0
#python eval_certified_densepure.py --max -1 --exp results/exp/cifar10 --config cifar10.yml --domain cifar10 --seed 0 --diffusion_type ddpm --lp_norm L2 --outfile results/ --sigma $SIGMA --N $N --N0 $N0 --certified_batch 100 --certify_mode purify --advanced_classifier cifar10-wideresnet --use_t_steps --num_t_steps $STEPS --save_predictions --predictions_path results/exp/ --reverse_seed=$REVSEED
#python eval_certified_densepure.py --max -1 --exp results/exp/cifar10 --config cifar10.yml --domain cifar10 --seed 0 --diffusion_type ddpm --lp_norm L2 --outfile results/ --sigma 1.0 --N 100000 --N0 100 --certified_batch 100 --certify_mode purify --advanced_classifier cifar10-wideresnet --use_t_steps --num_t_steps 10 --save_predictions --predictions_path results/exp/ --reverse_seed=0
#python eval_certified_densepure.py --max -1 --exp results/exp/cifar10 --config cifar10.yml --domain cifar10 --seed 0 --diffusion_type ddpm --lp_norm L2 --outfile results/ --sigma 0.25 --N 100 --N0 10 --certified_batch 100 --certify_mode purify --advanced_classifier cifar10-wideresnet --use_t_steps --num_t_steps 10 --save_predictions --predictions_path results/exp/ --reverse_seed=0

#python eval_certified_densepure.py --uap_test 1 --uap_target 0 --max -1 --skip 100 --exp results/exp/cifar10 --config cifar10.yml --domain cifar10 --seed 0 --diffusion_type ddpm --lp_norm L2 --outfile results/ --sigma 0.04 --N 100 --N0 10 --certified_batch 100 --certify_mode purify --advanced_classifier cifar10-wideresnet --use_t_steps --num_t_steps 10 --save_predictions --predictions_path results/exp/ --reverse_seed=0

python eval_certified_densepure.py --uap_test 1 --uap_target 0 --max -1 --skip 100 --exp results/exp/cifar10 --config cifar10.yml --domain cifar10 --seed 0 --diffusion_type ddpm --lp_norm L2 --outfile results/ --sigma 0.4 --N 100000 --N0 100 --certified_batch 100 --certify_mode purify --advanced_classifier cifar10-wideresnet --use_t_steps --num_t_steps 10 --save_predictions --predictions_path results/exp/ --reverse_seed=0



