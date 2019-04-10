# force matplotlib agg backend
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import numpy as np
import plot_evaluation as pe
import os
import sys
from argparse import ArgumentParser

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FILENAMES OF CLASSIFICATION FILES WITH APPROX. 500 PAIRS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dict_cos_by_model = { \
        "analogy": list(np.arange(0.1, 1.9 + 0.1, 0.1)), \
        "complex": list(np.arange(0.1, 1.9 + 0.1, 0.1)), \
        "distmult": list(np.arange(0.1, 2.8 + 0.1, 0.1)), \
        "hole": list(np.arange(0.1, 3.3 + 0.1, 0.1)), \
        "rescal": list(np.arange(0.1, 0.8 + 0.1, 0.1)), \
        "transd": list(np.arange(0.1, 2.5 + 0.1, 0.1)), \
        "transe": list(np.arange(0.1, 3.0 + 0.1, 0.1)), \
        "transh": list(np.arange(0.1, 2.9 + 0.1, 0.1)) \
        }

dict_l1_by_model = { \
        "analogy": list(np.arange(0.1, 2.6 + 0.1, 0.1)), \
        "complex": list(np.arange(0.1, 3.3 + 0.1, 0.1)), \
        "distmult": list(np.arange(0.1, 2.0 + 0.1, 0.1)), \
        "hole": list(np.arange(0.1, 4.1 + 0.1, 0.1)), \
        "rescal": list(np.arange(0.1, 4.4 + 0.1, 0.1)), \
        "transd": list(np.arange(0.1, 5.0 + 0.1, 0.1)), \
        "transe": list(np.arange(0.1, 3.8 + 0.1, 0.1)), \
        "transh": list(np.arange(0.1, 5.0 + 0.1, 0.1)) \
        }

FN_FORMATSTR = "experiments/" \
        + "dbpedia-201610N-1k-filtered_{0}/evaluation/" \
        + "class_{1:.2f}_percent_{2}_uris.txt"

# %%%%%%%%%%%%%
# READING FILES
# %%%%%%%%%%%%%

def get_pairs(fn, as_list=False):
    f = open(fn, "r")
    pairs = f.readlines()[1:]
    f.close()
    pairs = map(lambda x: frozenset(x.split()), pairs)
    if as_list:
        return list(pairs)
    else:
        return set(pairs)

# %%%%%%%%%%%%%%%%%%%%
# PRECISION "AT TOP K"
# %%%%%%%%%%%%%%%%%%%%

def get_true_positives(pairs, ground_truth_pairs, as_list=False):
    return set(filter(lambda x: x in ground_truth_pairs, pairs))

def precision(pairs, ground_truth_pairs):
    num_true_positives = len(get_true_positives(pairs, ground_truth_pairs))
    return (float(num_true_positives) / float(len(pairs))) if len(pairs) > 0 else 0.0

# %%%%%%%%%%
# EVALUATION
# %%%%%%%%%%

def save_evaluation(fn, values):
    f = open(fn, "w")
    f.writelines(map(lambda x: str(x[0]) + "\t" + str(x[1]) + "\n", values))
    f.flush()
    f.close()

def evaluate_baseline(baseline_pairs, ground_truth_pairs):
    # baseline pairs are sorted by confidence (descending)
    # calculate precision for first k pairs (0 < k < top k)
    values = []
    for k in range(1, min(len(baseline_pairs), len(ground_truth_pairs)) + 1):
        values.append([precision(baseline_pairs[:k], ground_truth_pairs), k])
    return values

def evaluate_dbpedia(fn_ground_truth_pairs, fn_baseline):
    # get ground-truth and baseline pairs first and prepare evaluation
    ground_truth_pairs = get_pairs(fn_ground_truth_pairs)
    baseline_pairs = get_pairs(fn_baseline, True)
    top_k = 500         # hard coded
    
    # evaluate baseline
    sys.stdout.write("Evaluating baseline ... ")
    sys.stdout.flush()
    values_baseline = evaluate_baseline(baseline_pairs, ground_truth_pairs)
    save_evaluation("evaluation_baseline.txt", values_baseline)
    print("Done")
    
    # evaluate every model (cos)
    values_by_model_cos = {}
    for model in dict_cos_by_model.keys():
        # evaluate every classification list of that model
        sys.stdout.write("Evaluating {0} (COS) ... ".format(model))
        sys.stdout.flush()
        values = []
        for threshold in dict_cos_by_model[model]:
            # load list
            fn = FN_FORMATSTR.format(model, threshold, "cos")
            pairs = get_pairs(fn)
            values.append([precision(pairs, ground_truth_pairs), len(pairs)])
            
            # if length of pairs equals or is greater than top k, stop
            if len(pairs) >= top_k:
                break
        # save values of evaluation for this model to final dict
        save_evaluation("evaluation_{0}_cos.txt".format(model), values)
        values_by_model_cos[model] = values
        print("Done")
    values_by_model_cos["baseline"] = values_baseline.copy()
    
    # plot evaluation (cos)
    sys.stdout.write("Plotting (COS) ... ")
    sys.stdout.flush()
    pe.plot_evaluation("dbpedia-201610N-1k-filtered_cos.pdf", "TOP K", "PRECISION", None, \
            xlim=[0, top_k], ylim=[0, 1.0], xtick_step=50, ytick_step=0.1, \
            **values_by_model_cos)
    print("Done")
    
    # evaluate every model (l1)
    values_by_model_l1 = {}
    for model in dict_l1_by_model.keys():
        # evaluate every classification list of that model
        sys.stdout.write("Evaluating {0} (L1) ... ".format(model))
        sys.stdout.flush()
        values = []
        for threshold in dict_l1_by_model[model]:
            # load list
            fn = FN_FORMATSTR.format(model, threshold, "l1")
            pairs = get_pairs(fn)
            values.append([precision(pairs, ground_truth_pairs), len(pairs)])

            # if length of pairs equals or is greater than top k, stop
            if len(pairs) >= top_k:
                break
        # save values of evaluation for this model to final dict
        save_evaluation("evaluation_{0}_l1.txt".format(model), values)
        values_by_model_l1[model] = values
        print("Done")
    values_by_model_l1["baseline"] = values_baseline.copy()

    # plot evaluation (l1)
    sys.stdout.write("Plotting (L1) ... ")
    sys.stdout.flush()
    pe.plot_evaluation("dbpedia-201610N-1k-filtered_l1.pdf", "TOP K", "PRECISION", None, \
            xlim=[0, top_k], ylim=[0, 1], xtick_step=50, ytick_step=0.1, \
            **values_by_model_l1)
    print("Done")

# %%%%%%%%%%%%%
# MAIN FUNCTION
# %%%%%%%%%%%%%

def main():
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("GOLD", type=str, \
            help="The ground-truth pairs file to use for evaluation.")
    parser.add_argument("BASELINE", type=str, \
            help="The baseline evaluation file to add to the plots.")
    parser.add_argument("-c", "--color-map", type=str, choices=plt.colormaps(), default="nipy_spectral_r", \
            help="The color map to use for plotting multiple curves in one diagram (Default: nipy_spectral_r).")
    parser.add_argument("-g", "--gaussian-enable", action="store_true", default=False, \
            help="Enable gaussian filter for smoothing (Default: False).")
    parser.add_argument("-s", "--gaussian-sigma", type=float, default=2.0, \
            help="The sigma scalar for gaussian filter kernel, use with -g option (Default: 2.0).")
    parser.add_argument("-i", "--interpolation-enable", action="store_true", default=False, \
            help="Enable interpolation for smoothing (Default: False).")
    parser.add_argument("-n", "--interpolation-segments", type=int, default=100, \
            help="The number of equidistant segments along recall axis for interpolation,"
            + " use with -i option (Default: 100).")
    parser.add_argument("-o", "--interpolation-order", type=int, default=15, \
            help="The interpolation order, use with -i option (Default: 15).")
    args = parser.parse_args()
    
    # set parameters
    pe.color_map = args.color_map
    pe.gaussian_enable = args.gaussian_enable
    pe.gaussian_sigma = args.gaussian_sigma
    pe.interpolation_enable = args.interpolation_enable
    pe.interpolation_segments = args.interpolation_segments
    pe.interpolation_order = args.interpolation_order
    
    # check filenames
    if not os.path.exists(args.GOLD):
        print("ERROR: ground-truth pairs not found.")
        sys.exit(-1)
    if not os.path.exists(args.BASELINE):
        print("ERROR: baseline pairs not found.")
        sys.exit(-1)
    
    # evaluate
    evaluate_dbpedia(args.GOLD, args.BASELINE)

if __name__ == "__main__":
    main()

