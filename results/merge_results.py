import argparse
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm

def gain_results(args):
    file_merge = open(str(args.datasets)+'_'+str(args.sigma)+'_'+str(args.results_file), 'w')
    file_merge.write("idx\tlabel\tpredict\tradius\tcorrect\n")
    if args.use_id:
        for sample_id in range(args.sample_num):

            n0_predictions_list = []
            n_predictions_list = []
            for i in range(args.majority_vote_num):
                id_file = open(str(args.datasets)+'-densepure-sample_num'+str(args.N)+'-noise_'+str(args.sigma)+'-'+str(args.steps)+'-steps-'+str(i), 'r')
                lines = id_file.readlines()
                line = lines[sample_id+1].split('\t')
                label = int(line[1])

                n0_predictions = np.load('exp/'+str(args.datasets)+'/'+str(args.sigma)+'-'+str(args.sample_id_list[sample_id])+'-'+str(i)+'-n0_predictions.npy')
                n_predictions = np.load('exp/'+str(args.datasets)+'/'+str(args.sigma)+'-'+str(args.sample_id_list[sample_id])+'-'+str(i)+'-n_predictions.npy')

                n0_predictions_list.append(n0_predictions)
                n_predictions_list.append(n_predictions)

            n0_predictions_list = np.array(n0_predictions_list).T
            n_predictions_list = np.array(n_predictions_list).T
            count_max_list = np.zeros(args.N0,dtype=int)

            for i in range(args.N0):
                count_max = max(list(n0_predictions_list[i]),key=list(n0_predictions_list[i]).count)
                count_max_list[i] = count_max
            counts = np.zeros(args.classes_num, dtype=int)
            for idx in count_max_list:
                counts[idx] += 1
            prediction = counts.argmax().item()

            count_max_list = np.zeros(args.N,dtype=int)
            for i in range(args.N):
                count_max = max(list(n_predictions_list[i]),key=list(n_predictions_list[i]).count)
                count_max_list[i] = count_max
            counts = np.zeros(args.classes_num, dtype=int)
            for idx in count_max_list:
                counts[idx] += 1

            nA = counts[prediction].item()
            pABar = proportion_confint(nA, args.N, alpha=2 * 0.001, method="beta")[0]
            if pABar < 0.5:
                prediction = -1
                radius = 0.0
            else:
                radius = args.sigma * norm.ppf(pABar)

            correct = int(prediction == label)

            file_merge.write("{}\t{}\t{}\t{:.3}\t{}".format(args.sample_id_list[sample_id], label, prediction, radius, correct))
            file_merge.write("\n")
    else:
        seq = 0
        for sample_id in range(args.max):

            # only certify every args.skip examples, and stop after args.max examples
            if sample_id % args.skip != 0:
                continue
            if sample_id == args.max:
                break

            n0_predictions_list = []
            n_predictions_list = []
            for i in range(args.majority_vote_num):
                id_file = open(str(args.datasets) + '-densepure-sample_num' + str(args.N) + '-noise_' + str(
                    args.sigma) + '-' + str(args.steps) + '-' + str(i), 'r')
                lines = id_file.readlines()
                line = lines[seq + 1].split('\t')
                label = int(line[1])

                n0_predictions = np.load('exp/' + str(args.datasets) + '/' + str(args.sigma) + '-' + str(
                    sample_id) + '-' + str(i) + '-n0_predictions.npy')
                n_predictions = np.load('exp/' + str(args.datasets) + '/' + str(args.sigma) + '-' + str(
                    sample_id) + '-' + str(i) + '-n_predictions.npy')

                n0_predictions_list.append(n0_predictions)
                n_predictions_list.append(n_predictions)

            n0_predictions_list = np.array(n0_predictions_list).T
            n_predictions_list = np.array(n_predictions_list).T
            count_max_list = np.zeros(args.N0, dtype=int)

            for i in range(args.N0):
                count_max = max(list(n0_predictions_list[i]), key=list(n0_predictions_list[i]).count)
                count_max_list[i] = count_max
            counts = np.zeros(args.classes_num, dtype=int)
            for idx in count_max_list:
                counts[idx] += 1
            prediction = counts.argmax().item()

            count_max_list = np.zeros(args.N, dtype=int)
            for i in range(args.N):
                count_max = max(list(n_predictions_list[i]), key=list(n_predictions_list[i]).count)
                count_max_list[i] = count_max
            counts = np.zeros(args.classes_num, dtype=int)
            for idx in count_max_list:
                counts[idx] += 1

            nA = counts[prediction].item()
            pABar = proportion_confint(nA, args.N, alpha=2 * 0.001, method="beta")[0]
            if pABar < 0.5:
                prediction = -1
                radius = 0.0
            else:
                radius = args.sigma * norm.ppf(pABar)

            correct = int(prediction == label)

            file_merge.write(
                "{}\t{}\t{}\t{:.3}\t{}".format(sample_id, label, prediction, radius, correct))
            file_merge.write("\n")

            seq = seq + 1

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--use_id', action='store_true', help='evaluate specific sample')
    parser.add_argument("--sample_id_list", type=int, nargs='+', default=[0], help="sample id for evaluation")
    parser.add_argument("--skip", type=int, default=100, help="how many examples to skip")
    parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
    parser.add_argument('--sample_num', type=int, default=100, help='sample numbers')
    parser.add_argument('--majority_vote_num', type=int, default=10, help='majority vote numbers')
    parser.add_argument("--N0", type=int, default=100)
    parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
    parser.add_argument('--sigma', type=float, default=0.25, help='noise hyperparameter')
    parser.add_argument('--classes_num', type=int, default=10, help='classes numbers of datasets')
    parser.add_argument("--results_file", type=str, default='merge_results.txt', help="output file")
    parser.add_argument("--datasets", type=str, default='cifar10', help="cifar10 or imagenet")
    parser.add_argument("--steps", type=int, default=2)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    print(args)
    gain_results(args)