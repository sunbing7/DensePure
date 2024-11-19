

for TARGET_CLASS in {0,1,2,3,5,4,6,7,8,9}
do
  python merge_result_ours.py --uap_target $TARGET_CLASS --sample_num 100 --max 10000 --skip 100 --majority_vote_num 40 --N 100 --N0 10 --sigma 0.25 --classes_num 10 --datasets cifar10 --steps 10
done
#python merge_result_ours.py --uap_target 0 --sample_num 100 --max 10000 --skip 100 --majority_vote_num 40 --N 100 --N0 10 --sigma 0.25 --classes_num 10 --datasets cifar10 --steps 10