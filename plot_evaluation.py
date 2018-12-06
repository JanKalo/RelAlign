#!/usr/bin/python -u

# force matplotlib agg backend
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import os
from argparse import ArgumentParser

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# PLOTTING VALUES OF MULTIPLE MODELS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_evaluation(file_name, xlabel, ylabel, title, dynamic_scale=False, **values_by_model):
    # prepare plotting
    get_precision = lambda x: list(map(lambda y: float(y[0]), x))
    get_recall = lambda x: list(map(lambda y: float(y[1]), x))
    get_valid_num = lambda x: len(list(filter(lambda y: y > 0.0, x)))
    plt.clf()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    if dynamic_scale:
        max_val_x = max(map(lambda x: x[1], values_by_model.values()))
        max_val_y = max(map(lambda x: x[0], values_by_model.values()))
        max_val_x += max_val_x / 10.0
        max_val_y += max_val_y / 10.0
        plt.xlim([0, max_val_x if max_val_x > 0.0 and max_val_x < 1.0 else 1])
        plt.ylim([0, max_val_y if max_val_y > 0.0 and max_val_y < 1.0 else 1])

    # plot rescal
    if "rescal" in values_by_model:
        precision = get_precision(values_by_model["rescal"])
        recall = get_recall(values_by_model["rescal"])
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], "r-", label = "RESCAL")
    # plot transe
    if "transe" in values_by_model:
        precision = get_precision(values_by_model["transe"])
        recall = get_recall(values_by_model["transe"])
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], "g-", label = "TransE")
    # plot transh
    if "transh" in values_by_model:
        precision = get_precision(values_by_model["transh"])
        recall = get_recall(values_by_model["transh"])
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], "b-", label = "TransH")
    # plot transd
    if "transd" in values_by_model:
        precision = get_precision(values_by_model["transd"])
        recall = get_recall(values_by_model["transd"])
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], "k-", label = "TransD")
    # plot analogy
    if "analogy" in values_by_model:
        precision = get_precision(values_by_model["analogy"])
        recall = get_recall(values_by_model["analogy"])
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], "m-", label = "Analogy")
    # plot baseline
    if "baseline" in values_by_model:
        precision = get_precision(values_by_model["baseline"])
        recall = get_recall(values_by_model["baseline"])
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], "y-", label = "Baseline")
    
    # label and save
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc = "upper right")
    plt.savefig(file_name)

def plot_experiment_evaluation(experiment_dir_without_model_suffix, \
        dynamic_scale=False, print_title=False, baseline_fn=None):
    def read_evaluation(file_name):
        evaluation_file = open(file_name, "r")
        values = list(map(lambda x: x.split(), evaluation_file.readlines()))
        evaluation_file.close()
        return values
    
    # check which models exist
    experiment = experiment_dir_without_model_suffix.rstrip("/")
    dirname = os.path.dirname(experiment)
    basename = os.path.basename(experiment)
    rescal_fn = experiment + "_rescal"
    transe_fn = experiment + "_transe"
    transh_fn = experiment + "_transh"
    transd_fn = experiment + "_transd"
    analogy_fn = experiment + "_analogy"
    rescal = os.path.exists(rescal_fn)
    transe = os.path.exists(transe_fn)
    transh = os.path.exists(transh_fn)
    transd = os.path.exists(transd_fn)
    analogy = os.path.exists(analogy_fn)
    
    # get baseline values if specified
    baseline = baseline_fn and os.path.exists(baseline_fn)
    baseline_values = read_evaluation(baseline_fn) if baseline else None

    # l1 distance plot
    rescal_values = read_evaluation(os.path.join(rescal_fn, \
            "evaluation", "evaluation_class_l1.txt")) if rescal else None
    transe_values = read_evaluation(os.path.join(transe_fn, \
            "evaluation", "evaluation_class_l1.txt")) if transe else None
    transh_values = read_evaluation(os.path.join(transh_fn, \
            "evaluation", "evaluation_class_l1.txt")) if transh else None
    transd_values = read_evaluation(os.path.join(transd_fn, \
            "evaluation", "evaluation_class_l1.txt")) if transd else None
    analogy_values = read_evaluation(os.path.join(analogy_fn, \
            "evaluation", "evaluation_class_l1.txt")) if analogy else None
    file_name = os.path.join(dirname, basename + "_l1.pdf")
    title = basename + ": Evaluation (L1-norm distance)" \
            if print_title else None
    values_by_model = {}
    if rescal:
        values_by_model["rescal"] = rescal_values
    if transe:
        values_by_model["transe"] = transe_values
    if transh:
        values_by_model["transh"] = transh_values
    if transd:
        values_by_model["transd"] = transd_values
    if analogy:
        values_by_model["analogy"] = analogy_values
    if baseline:
        values_by_model["baseline"] = baseline_values
    plot_evaluation(file_name, "RECALL", "PRECISION", title, dynamic_scale, **values_by_model)

    # cosine similarity plot
    rescal_values = read_evaluation(os.path.join(rescal_fn, \
            "evaluation", "evaluation_class_cos.txt")) if rescal else None
    transe_values = read_evaluation(os.path.join(transe_fn, \
            "evaluation", "evaluation_class_cos.txt")) if transe else None
    transh_values = read_evaluation(os.path.join(transh_fn, \
            "evaluation", "evaluation_class_cos.txt")) if transh else None
    transd_values = read_evaluation(os.path.join(transd_fn, \
            "evaluation", "evaluation_class_cos.txt")) if transd else None
    analogy_values = read_evaluation(os.path.join(analogy_fn, \
            "evaluation", "evaluation_class_cos.txt")) if analogy else None
    file_name = os.path.join(dirname, basename + "_cos.pdf")
    title = basename + ": Evaluation (cosine similarity)" \
            if print_title else None
    values_by_model = {}
    if rescal:
        values_by_model["rescal"] = rescal_values
    if transe:
        values_by_model["transe"] = transe_values
    if transh:
        values_by_model["transh"] = transh_values
    if transd:
        values_by_model["transd"] = transd_values
    if analogy:
        values_by_model["analogy"] = analogy_values
    if baseline:
        values_by_model["baseline"] = baseline_values
    plot_evaluation(file_name, "RECALL", "PRECISION", title, dynamic_scale, **values_by_model)

def plot_fb15k_evaluation(dynamic_scale=False, print_title=False):
    # for every fb15k embedding
    fb15k_names = "FB15K_{0}_{1}"
    experiments_fn = "experiments/tf-gpu_1.11.0/"
    percentages = range(10, 60, 10)
    min_occurences = range(200, 2200, 200)
    for percentage in percentages:
        for min_occurence in min_occurences:
            fb15k_name = fb15k_names.format(percentage, min_occurence)
            plot_experiment_evaluation(os.path.join(experiments_fn, fb15k_name), \
                    dynamic_scale, print_title)

def main():
    # parse arguments
    parser = ArgumentParser()
    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument("-f", "--fb15k-analysis", action="store_true", \
            default=False, help="Perform plotting for all FB15K experiments")
    exclusive_group.add_argument("-e", "--experiment", type=str, default=None, \
            help="Plot evaluations of every model for this experiment" \
            + " (directory without model suffix, so this script can" \
            + " get every model to evaluate). If None (not specified)," \
            + " it will evaluate each FB15K experiment (without any baseline)" \
            + " (Default: None).")
    parser.add_argument("-b", "--baseline", type=str, default=None, \
            help="An optional baseline evaluation file to add to the plots" \
            + " of a specified experiment (ignored with -f) (Default: None)")
    parser.add_argument("-d", "--dynamic_scale", action="store_true", default=False, \
            help="Scale plots dynamically by values (Default: False)")
    parser.add_argument("-t", "--print-title", action="store_true", default=False, \
            help="Print titles in plots (Default: False)")
    args = parser.parse_args()

    # plot for experiment if argument specified,
    # else just plot for fb15k if -f specified
    if args.fb15k_analysis:
        print("Plotting for all FB15K experiments")
        plot_fb15k_evaluation(args.dynamic_scale, args.print_title)
    elif args.experiment:
        print("Plotting for " + args.experiment)
        plot_experiment_evaluation(args.experiment, \
                args.dynamic_scale, args.print_title, args.baseline)

if __name__ == "__main__":
    main()

