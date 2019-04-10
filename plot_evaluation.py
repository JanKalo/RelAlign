#!/usr/bin/python -u

# force matplotlib agg backend
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

import os
import numpy as np
from argparse import ArgumentParser

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# PLOTTING-RELATED TWEAKABLES
# %%%%%%%%%%%%%%%%%%%%%%%%%%%

color_map = None
print_title = None
gaussian_enable = None
gaussian_sigma = None
interpolation_enable = None
interpolation_segments = None
interpolation_order = None

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# PLOTTING VALUES OF MULTIPLE MODELS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_evaluation(file_name, xlabel, ylabel, title, \
        xlim=[0.0, 1.0], ylim=[0.0, 1.0], xtick_step=0.1, ytick_step=0.1, \
        **values_by_model):
    # prepare plotting
    get_precision = lambda x: list(map(lambda y: float(y[0]), x))
    get_recall = lambda x: list(map(lambda y: float(y[1]), x))
    get_valid_num = lambda x: len(list(filter(lambda y: y > 0.0, x)))
    plt.clf()
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    # get color map
    cmap = plt.get_cmap(color_map)
    num_colors = len(values_by_model.keys())
    def get_color(n):
        return cmap(float(n) * (1.0 / num_colors))
    
    # plot rescal
    if "rescal" in values_by_model:
        precision = get_precision(values_by_model["rescal"])
        recall = get_recall(values_by_model["rescal"])
        if gaussian_enable:
            precision = gaussian_filter1d(precision, sigma=gaussian_sigma)
        if interpolation_enable:
            new_recall = np.linspace(min(recall), max(recall), interpolation_segments)
            f = UnivariateSpline(recall, precision, k=interpolation_order)
            precision = f(new_recall)
            recall = new_recall
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], \
                color=get_color(1), linestyle='solid', label = "RESCAL")
    # plot transe
    if "transe" in values_by_model:
        precision = get_precision(values_by_model["transe"])
        recall = get_recall(values_by_model["transe"])
        if gaussian_enable:
            precision = gaussian_filter1d(precision, sigma=gaussian_sigma)
        if interpolation_enable:
            new_recall = np.linspace(min(recall), max(recall), interpolation_segments)
            f = UnivariateSpline(recall, precision, k=interpolation_order)
            precision = f(new_recall)
            recall = new_recall
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], \
                color=get_color(2), linestyle='solid', label = "TransE")
    # plot transh
    if "transh" in values_by_model:
        precision = get_precision(values_by_model["transh"])
        recall = get_recall(values_by_model["transh"])
        if gaussian_enable:
            precision = gaussian_filter1d(precision, sigma=gaussian_sigma)
        if interpolation_enable:
            new_recall = np.linspace(min(recall), max(recall), interpolation_segments)
            f = UnivariateSpline(recall, precision, k=interpolation_order)
            precision = f(new_recall)
            recall = new_recall
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], \
                color=get_color(3), linestyle='solid', label = "TransH")
    # plot transd
    if "transd" in values_by_model:
        precision = get_precision(values_by_model["transd"])
        recall = get_recall(values_by_model["transd"])
        if gaussian_enable:
            precision = gaussian_filter1d(precision, sigma=gaussian_sigma)
        if interpolation_enable:
            new_recall = np.linspace(min(recall), max(recall), interpolation_segments)
            f = UnivariateSpline(recall, precision, k=interpolation_order)
            precision = f(new_recall)
            recall = new_recall
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], \
                color=get_color(4), linestyle='solid', label = "TransD")
    # plot distmult
    if "distmult" in values_by_model:
        precision = get_precision(values_by_model["distmult"])
        recall = get_recall(values_by_model["distmult"])
        if gaussian_enable:
            precision = gaussian_filter1d(precision, sigma=gaussian_sigma)
        if interpolation_enable:
            new_recall = np.linspace(min(recall), max(recall), interpolation_segments)
            f = UnivariateSpline(recall, precision, k=interpolation_order)
            precision = f(new_recall)
            recall = new_recall
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], \
                color=get_color(5), linestyle='solid', label = "DistMult")
    # plot hole
    if "hole" in values_by_model:
        precision = get_precision(values_by_model["hole"])
        recall = get_recall(values_by_model["hole"])
        if gaussian_enable:
            precision = gaussian_filter1d(precision, sigma=gaussian_sigma)
        if interpolation_enable:
            new_recall = np.linspace(min(recall), max(recall), interpolation_segments)
            f = UnivariateSpline(recall, precision, k=interpolation_order)
            precision = f(new_recall)
            recall = new_recall
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], \
                color=get_color(6), linestyle='solid', label = "HolE")
    # plot complex
    if "complex" in values_by_model:
        precision = get_precision(values_by_model["complex"])
        recall = get_recall(values_by_model["complex"])
        if gaussian_enable:
            precision = gaussian_filter1d(precision, sigma=gaussian_sigma)
        if interpolation_enable:
            new_recall = np.linspace(min(recall), max(recall), interpolation_segments)
            f = UnivariateSpline(recall, precision, k=interpolation_order)
            precision = f(new_recall)
            recall = new_recall
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], \
                color=get_color(7), linestyle='solid', label = "ComplEx")
    # plot analogy
    if "analogy" in values_by_model:
        precision = get_precision(values_by_model["analogy"])
        recall = get_recall(values_by_model["analogy"])
        if gaussian_enable:
            precision = gaussian_filter1d(precision, sigma=gaussian_sigma)
        if interpolation_enable:
            new_recall = np.linspace(min(recall), max(recall), interpolation_segments)
            f = UnivariateSpline(recall, precision, k=interpolation_order)
            precision = f(new_recall)
            recall = new_recall
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], \
                color=get_color(8), linestyle='solid', label = "Analogy")
    # plot baseline
    if "baseline" in values_by_model:
        precision = get_precision(values_by_model["baseline"])
        recall = get_recall(values_by_model["baseline"])
        if gaussian_enable:
            precision = gaussian_filter1d(precision, sigma=gaussian_sigma)
        if interpolation_enable:
            new_recall = np.linspace(min(recall), max(recall), interpolation_segments)
            f = UnivariateSpline(recall, precision, k=interpolation_order)
            precision = f(new_recall)
            recall = new_recall
        min_idx = len(precision) - min(get_valid_num(precision), get_valid_num(recall))
        plt.plot(recall[min_idx:], precision[min_idx:], \
                color=get_color(9), linestyle='dashed', label = "Baseline")
    
    # tick frequency
    axes = plt.gca()
    start, end = axes.get_xlim()
    axes.set_xticks(np.arange(start, end, xtick_step))
    start, end = axes.get_ylim()
    axes.set_yticks(np.arange(start, end, ytick_step))
    
    # show grid
    plt.grid(True)
    
    # label and save
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc = "upper right")
    plt.savefig(file_name)

def plot_experiment_evaluation(experiment_dir_without_model_suffix, baseline_fn=None):
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
    distmult_fn = experiment + "_distmult"
    hole_fn = experiment + "_hole"
    complex_fn = experiment + "_complex"
    analogy_fn = experiment + "_analogy"
    rescal = os.path.exists(rescal_fn)
    transe = os.path.exists(transe_fn)
    transh = os.path.exists(transh_fn)
    transd = os.path.exists(transd_fn)
    distmult = os.path.exists(distmult_fn)
    hole = os.path.exists(hole_fn)
    complex = os.path.exists(complex_fn)
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
    distmult_values = read_evaluation(os.path.join(distmult_fn, \
            "evaluation", "evaluation_class_l1.txt")) if distmult else None
    hole_values = read_evaluation(os.path.join(hole_fn, \
            "evaluation", "evaluation_class_l1.txt")) if hole else None
    complex_values = read_evaluation(os.path.join(complex_fn, \
            "evaluation", "evaluation_class_l1.txt")) if complex else None
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
    if distmult:
        values_by_model["distmult"] = distmult_values
    if hole:
        values_by_model["hole"] = hole_values
    if complex:
        values_by_model["complex"] = complex_values
    if analogy:
        values_by_model["analogy"] = analogy_values
    if baseline:
        values_by_model["baseline"] = baseline_values
    plot_evaluation(file_name, "RECALL", "PRECISION", title, **values_by_model)

    # cosine similarity plot
    rescal_values = read_evaluation(os.path.join(rescal_fn, \
            "evaluation", "evaluation_class_cos.txt")) if rescal else None
    transe_values = read_evaluation(os.path.join(transe_fn, \
            "evaluation", "evaluation_class_cos.txt")) if transe else None
    transh_values = read_evaluation(os.path.join(transh_fn, \
            "evaluation", "evaluation_class_cos.txt")) if transh else None
    transd_values = read_evaluation(os.path.join(transd_fn, \
            "evaluation", "evaluation_class_cos.txt")) if transd else None
    distmult_values = read_evaluation(os.path.join(distmult_fn, \
            "evaluation", "evaluation_class_cos.txt")) if distmult else None
    hole_values = read_evaluation(os.path.join(hole_fn, \
            "evaluation", "evaluation_class_cos.txt")) if hole else None
    complex_values = read_evaluation(os.path.join(complex_fn, \
            "evaluation", "evaluation_class_cos.txt")) if complex else None
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
    if distmult:
        values_by_model["distmult"] = distmult_values
    if hole:
        values_by_model["hole"] = hole_values
    if complex:
        values_by_model["complex"] = complex_values
    if analogy:
        values_by_model["analogy"] = analogy_values
    if baseline:
        values_by_model["baseline"] = baseline_values
    plot_evaluation(file_name, "RECALL", "PRECISION", title, **values_by_model)

def plot_fb15k_evaluation():
    # for every fb15k embedding
    fb15k_names = "FB15K_{0}_{1}"
    experiments_fn = "experiments/tf-gpu_1.11.0/"
    percentages = range(10, 60, 10)
    min_occurences = range(200, 2200, 200)
    for percentage in percentages:
        for min_occurence in min_occurences:
            fb15k_name = fb15k_names.format(percentage, min_occurence)
            plot_experiment_evaluation(os.path.join(experiments_fn, fb15k_name))

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
            + " of a specified experiment (ignored with -f) (Default: None).")
    parser.add_argument("-c", "--color-map", type=str, choices=plt.colormaps(), default="nipy_spectral_r", \
            help="The color map to use for plotting multiple curves in one diagram (Default: nipy_spectral_r).")
    parser.add_argument("-t", "--print-title", action="store_true", default=False, \
            help="Print titles in plots (Default: False).")
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
    global color_map
    global print_title
    global gaussian_enable
    global gaussian_sigma
    global interpolation_enable
    global interpolation_segments
    global interpolation_order
    color_map = args.color_map
    print_title = args.print_title
    gaussian_enable = args.gaussian_enable
    gaussian_sigma = args.gaussian_sigma
    interpolation_enable = args.interpolation_enable
    interpolation_segments = args.interpolation_segments
    interpolation_order = args.interpolation_order
    
    # plot for experiment if argument specified,
    # else just plot for fb15k if -f specified
    if args.fb15k_analysis:
        print("Plotting for all FB15K experiments")
        plot_fb15k_evaluation()
    elif args.experiment:
        print("Plotting for " + args.experiment)
        plot_experiment_evaluation(args.experiment, args.baseline)

if __name__ == "__main__":
    main()

