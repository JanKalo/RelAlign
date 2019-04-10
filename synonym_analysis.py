# force matplotlib agg backend
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from thirdParty.OpenKE import models
from embedding import Embedding
import numpy as np
import pandas as pd
import os
import sys

# %%%%%%%%
# PLOTTING
# %%%%%%%%

def csv_to_plot(csv_fn, file_name, xlabel, ylabel, title):
    series = pd.read_csv(csv_fn, squeeze=True, index_col=0, encoding="utf-8")
    plt.clf()
    series.plot.line()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(file_name)

#########################################
# MODELSPECIFIC RELATION MEASURES delta_r
#########################################

def get_rescal_delta_r(embedding_params, embedding_dimension, rel_idx):
    # get matrix embedding R_rel_idx from "rel_matrices" parameter
    R_rel_idx = embedding_params["rel_matrices"][rel_idx]

    # R_rel_idx shaped as vector with dimension d * d is the measure
    delta_r = R_rel_idx
    return delta_r

def get_transe_delta_r(embedding_params, embedding_dimension, rel_idx):
    # get vector embedding r_rel_idx from "rel_embeddings" parameter
    r_rel_idx = embedding_params["rel_embeddings"][rel_idx]

    # r_rel_idx is the measure
    delta_r = r_rel_idx
    return delta_r

def get_transh_delta_r(embedding_params, embedding_dimension, rel_idx):
    # ---- OLD ----
    # get vector embedding d_r_rel_idx from "rel_embeddings" parameter
    # get vector embedding w_r_rel_idx from "normal_vectors" parameter
    #d_r_rel_idx = embedding_params["rel_embeddings"][rel_idx]
    #w_r_rel_idx = embedding_params["normal_vectors"][rel_idx]

    # ---- OLD ----
    # d_r_rel_idx projected to w_r_rel_idx hyperplane is the measure
    #delta_r = np.subtract(d_r_rel_idx,
    #        np.multiply(np.sum(np.multiply(d_r_rel_idx, w_r_rel_idx)), w_r_rel_idx))
    # -------------

    # d_r_rel_idx concatenated with w_r_rel_idx is the measure
    #delta_r = np.concatenate((d_r_rel_idx, w_r_rel_idx))
    # -------------

    # measure is the same as in TransE, even stored in the same parameter
    delta_r = get_transe_delta_r(embedding_params, embedding_dimension, rel_idx)
    return delta_r

def get_transr_delta_r(embedding_params, embedding_dimension, rel_idx):
    # get vector embedding r_rel_idx from "rel_embeddings" parameter
    # get matrix embedding M_r_rel_idx from "transfer_matrix" parameter
    r_rel_idx = np.reshape(embedding_params["rel_embeddings"][rel_idx],
            (-1, embedding_dimension, 1))
    M_r_rel_idx = np.reshape(embedding_params["transfer_matrix"][rel_idx],
            (embedding_dimension, embedding_dimension))

    # r_rel_idx projected with M_r_rel_idx is the measure
    delta_r = np.reshape(np.matmul(M_r_rel_idx, r_rel_idx), embedding_dimension)
    return delta_r

def get_transd_delta_r(embedding_params, embedding_dimension, rel_idx):
    # get vector embedding r_rel_idx from "rel_embeddings" parameter
    # get vector embedding r_p_rel_idx from "rel_transfer" parameter
    r_rel_idx = embedding_params["rel_embeddings"][rel_idx]
    r_p_rel_idx = embedding_params["rel_transfer"][rel_idx]

    # ---- OLD ----
    # calculate projection matrix for r_rel_idx projection
    #M_r_rel_idx = np.add(np.multiply(r_p_rel_idx, np.array([[1] * embedding_dimension])), np.identity(embedding_dimension))

    # r_rel_idx projected with M_r_rel_idx is the measure
    #delta_r = np.matmul(M_r_rel_idx, r_rel_idx)
    #delta_r = np.add(r_rel_idx,
    #        np.multiply(np.sum(r_rel_idx), r_p_rel_idx))
    # -------------

    # r_rel_idx concatenated with r_p_rel_idx is the measure
    delta_r = np.concatenate((r_rel_idx, r_p_rel_idx))
    return delta_r

def get_distmult_delta_r(embedding_params, embedding_dimension, rel_idx):
    # measure is the same as in TransE, even stored in the same parameter
    delta_r = get_transe_delta_r(embedding_params, embedding_dimension, rel_idx)
    return delta_r

def get_hole_delta_r(embedding_params, embedding_dimension, rel_idx):
    # measure is the same as in TransE, even stored in the same parameter
    delta_r = get_transe_delta_r(embedding_params, embedding_dimension, rel_idx)
    return delta_r

def get_complex_delta_r(embedding_params, embedding_dimension, rel_idx):
    # get real part vector embedding re_rel_idx from "rel_re_embeddings" parameter
    re_rel_idx = embedding_params["rel_re_embeddings"][rel_idx]

    # get imaginary part vector embedding ri_rel_idx from "rel_im_embeddings" parameter
    ri_rel_idx = embedding_params["rel_im_embeddings"][rel_idx]

    # real part concatenated with imaginary part is the measure
    delta_r = np.concatenate((re_rel_idx, ri_rel_idx))
    return delta_r

def get_analogy_delta_r(embedding_params, embedding_dimension, rel_idx):
    # get vector embedding r_rel_idx from "rel_embeddings" parameter
    r_rel_idx = embedding_params["rel_embeddings"][rel_idx]

    # get real part vector embedding re_rel_idx from "rel_re_embeddings" parameter
    re_rel_idx = embedding_params["rel_re_embeddings"][rel_idx]

    # get imaginary part vector embedding ri_rel_idx from "rel_im_embeddings" parameter
    ri_rel_idx = embedding_params["rel_im_embeddings"][rel_idx]

    # all of them concatenated is the measure
    delta_r = np.concatenate((r_rel_idx, re_rel_idx, ri_rel_idx))
    return delta_r

##############################
# SIMILARITY FUNCTIONS DELTA_r
##############################

def norm_DELTA_r(delta_r_i, delta_r_j, ord=2):
    dissim = np.subtract(delta_r_i, delta_r_j)
    DELTA_r = np.linalg.norm(dissim, ord=ord)
    return DELTA_r

def cos_DELTA_r(delta_r_i, delta_r_j, ord=2):
    dot_ij = np.dot(delta_r_i, delta_r_j)
    len_i = np.linalg.norm(delta_r_i, ord=2)
    len_j = np.linalg.norm(delta_r_j, ord=2)
    DELTA_r = dot_ij / (len_i * len_j)
    return DELTA_r

############################
# CREATING SIMILARITY MATRIX
############################

def calc_df_rel_mat_similarity(embedding, func_delta_r, func_DELTA_r, ord=2):
    str_info = "Calculating Similarity Matrix ... "
    sys.stdout.write(str_info)
    sys.stdout.flush()
    step = 0
    count = embedding.con.relTotal
    emb_params = embedding.con.get_parameters()
    emb_dimension = embedding.embedding_dimensions
    mat_similarity = []
    # go through every relation i and save similarities to j's into a vector tuple
    for i in range(count):
        delta_r_i = func_delta_r(emb_params, emb_dimension, i)
        vec_similarity = []
        for j in range(count):
            delta_r_j = func_delta_r(emb_params, emb_dimension, j)
            DELTA_r = func_DELTA_r(delta_r_i, delta_r_j, ord=ord)
            vec_similarity.append(DELTA_r)
        mat_similarity.append(vec_similarity)
        step += 1
        sys.stdout.write("\r" + str_info + "%d%%" % int(step * 100 / count))
        sys.stdout.flush()
    df_rel_mat_similarity = pd.DataFrame(mat_similarity)
    print("\r" + str_info + "Done")
    return df_rel_mat_similarity

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FIRST STEP TO CLASSIFICATION / THRESHOLDING: GETTING NEAREST NEIGHBORS FOR EVERY RELATION
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_df_rel_nearest_neighbors(embedding, df_rel_similarity_matrix, left_side=True):
    str_info = "Creating nearest neighbors dataframe ... "
    sys.stdout.write(str_info)
    sys.stdout.flush()
    nearest_neighbors = []
    nearest_neighbors_ids = []
    max_progress = len(df_rel_similarity_matrix.index)
    old_progress = -1
    for rel_id in range(max_progress):
        # progress output
        progress = int(rel_id * 100 / max_progress)
        if progress != old_progress:
            sys.stdout.write("\r" + str_info + "%d%%" % progress)
            sys.stdout.flush()
            old_progress = progress
        
        # get nearest neighbors list and check if this pair was already added
        series_rel_sorted_similarities = df_rel_similarity_matrix.iloc[rel_id].sort_values()
        rel_nn_id = int(series_rel_sorted_similarities.index[1 if left_side else -2])
        if [rel_id, rel_nn_id] in nearest_neighbors_ids:
            continue
        else:
            nearest_neighbors_ids.append([rel_id, rel_nn_id])
            nearest_neighbors_ids.append([rel_nn_id, rel_id])
        
        # this pair was not added before, add it
        nn_similarity = series_rel_sorted_similarities.iloc[1]
        rel = embedding.lookup_relation(rel_id)
        rel_nn = embedding.lookup_relation(rel_nn_id)
        rel_sim_min = series_rel_sorted_similarities.min()
        rel_sim_max = series_rel_sorted_similarities.max()
        rel_sim_mean = series_rel_sorted_similarities.mean()
        rel_sim_std = series_rel_sorted_similarities.std()
        series_rel_nn_similarities = df_rel_similarity_matrix.iloc[rel_nn_id]
        rel_nn_sim_min = series_rel_nn_similarities.min()
        rel_nn_sim_max = series_rel_nn_similarities.max()
        rel_nn_sim_mean = series_rel_nn_similarities.mean()
        rel_nn_sim_std = series_rel_nn_similarities.std()
        nearest_neighbors.append([rel_id, rel_nn_id, rel, rel_nn, nn_similarity,
            rel_sim_min, rel_sim_max, rel_sim_mean, rel_sim_std,
            rel_nn_sim_min, rel_nn_sim_max, rel_nn_sim_mean, rel_nn_sim_std])
    
    # create dataframe and return it
    df_nearest_neighbors = pd.DataFrame(nearest_neighbors,
        columns = ["rel_id", "rel_nn_id", "rel", "rel_nn", "nn_similarity",
            "rel_sim_min", "rel_sim_max", "rel_sim_mean", "rel_sim_std",
            "rel_nn_sim_min", "rel_nn_sim_max", "rel_nn_sim_mean", "rel_nn_sim_std"])
    print("\r" + str_info + "Done")
    return df_nearest_neighbors

######################################################################################################
# CLASSIFICATION: FOR A SERIES OF THE DISTRIBUTION OF THE DISTANCES FROM ONE RELATION TO ALL: Z-SCORE!
######################################################################################################

def z_score(observed_value, relation_dist_mean, relation_dist_std):
    return (observed_value - relation_dist_mean) / relation_dist_std

def z_score_threshold_chebyshevs_ineq(max_population_percent):
    return np.sqrt(1 / (max_population_percent / 2))

def classification(df_rel_mat_similarity, max_population_percent, left_side=True):
    classified_synonyms = set()
    z_score_threshold = z_score_threshold_chebyshevs_ineq(max_population_percent)
    for rel_i, series_relation_dist in df_rel_mat_similarity.iterrows():
        rel_i = int(rel_i)
        rel_i_dist_mean = series_relation_dist.mean()
        rel_i_dist_std = series_relation_dist.std()
        series_classified = series_relation_dist[
                z_score(series_relation_dist, rel_i_dist_mean, rel_i_dist_std) < -z_score_threshold if left_side else \
                        z_score(series_relation_dist, rel_i_dist_mean, rel_i_dist_std) > z_score_threshold]
        for rel_j, observed_dist_ij in series_classified.iteritems():
            pair = frozenset([rel_i, int(rel_j)])
            if len(pair) == 2 and pair not in classified_synonyms:
                classified_synonyms.add(pair)
    return classified_synonyms

# unused
def classification_static(df_rel_mat_similarity, cos_value_threshold):
    classified_synonyms = set()
    for rel_i, series_relation_dist in df_rel_mat_similarity.iterrows():
        rel_i = int(rel_i)
        series_classified = series_relation_dist[series_relation_dist > cos_value_threshold]
        for rel_j, observed_dist_ij in series_classified.iteritems():
            pair = frozenset([rel_i, int(rel_j)])
            if len(pair) == 2 and pair not in classified_synonyms:
                classified_synonyms.add(pair)
    return classified_synonyms

def nearest_neighbors(df_rel_mat_similarity, left_side=True):
    nearest_neighbors = set()
    for rel_i, series_relation_dist in df_rel_mat_similarity.iterrows():
        rel_i = int(rel_i)
        rel_j = int(series_relation_dist.sort_values().index[1 if left_side else -2])
        pair = frozenset([rel_i, int(rel_j)])
        if len(pair) == 2 and pair not in nearest_neighbors:
            nearest_neighbors.add(pair)
    return nearest_neighbors

####################################################################################
# CALCULATE PRECISION / RECALL BETWEEN CLASSIFIED SYNONYMS AND GROUND-TRUTH SYNONYMS
####################################################################################

def save_synonyms_set(synonyms_id_set, file_name_without_extension, embedding):
    # save ids
    synonyms_id_file = open(file_name_without_extension + "_ids.txt", "w")
    synonyms_id_file.write("%d\n" % len(synonyms_id_set))
    synonyms_id_file.writelines(map(lambda x: str(sorted(x)[0]) + "\t" + str(sorted(x)[1]) + "\n", list(synonyms_id_set)))
    synonyms_id_file.close()
    # if we have an embedding instance, also save uris as separate file
    if embedding:
        # create synonyms list with uris
        synonyms_uri_list = list(map(lambda x: \
                [embedding.lookup_relation(sorted(x)[0]), embedding.lookup_relation(sorted(x)[1])], \
                list(synonyms_id_set)))
        # save uris
        synonyms_uri_file = open(file_name_without_extension + "_uris.txt", "w")
        synonyms_uri_file.write("%d\n" % len(synonyms_uri_list))
        synonyms_uri_file.writelines(map(lambda x: str(x[0]) + "\t" + str(x[1]) + "\n", synonyms_uri_list))
        synonyms_uri_file.close()

def get_synonyms_set(synonyms_fn):
    synonyms_file = open(synonyms_fn, "r")
    synonyms_lines = synonyms_file.readlines()[1:]
    synonyms_set = set(map(lambda x: frozenset(map(lambda y: int(y), x.split())), synonyms_lines))
    synonyms_file.close()
    return synonyms_set

def calc_true_positives(classified_synonyms_set, ground_truth_synonyms_set):
    return set(filter(lambda x: x in ground_truth_synonyms_set, classified_synonyms_set))

def calc_precision_recall(classified_synonyms_set, ground_truth_synonyms_set):
    num_true_positives = len(calc_true_positives(classified_synonyms_set, ground_truth_synonyms_set))
    num_classified_positives = len(classified_synonyms_set)
    num_ground_truth_positives = len(ground_truth_synonyms_set)
    precision = (float(num_true_positives) / float(num_classified_positives)) if num_classified_positives != 0 else 0.0
    recall = (float(num_true_positives) / float(num_ground_truth_positives)) if num_ground_truth_positives != 0 else 0.0
    return precision, recall

# %%%%%%%%%%%%%%%%
# SYNONYM ANALYSIS
# %%%%%%%%%%%%%%%%

def prepare_output_dir(embedding, output_dir):
    # check if directories exist
    output_dir = output_dir.rstrip("/")
    sys.stdout.write("Preparing output directory \"%s\" ... " % output_dir)
    sys.stdout.flush()
    plots_dir = os.path.join(output_dir, "plots")
    evaluation_dir = os.path.join(output_dir, "evaluation")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(plots_dir)
        os.mkdir(evaluation_dir)
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    if not os.path.exists(evaluation_dir):
        os.mkdir(evaluation_dir)
    print("Done")

    # check if scalar plots exist, create if not
    sys.stdout.write("Plotting Scalars if not done yet ... ")
    sys.stdout.flush()
    plotted = False
    csv_loss_fn = os.path.join(embedding.embedding_dir, "loss.csv")
    plot_training_fn = os.path.join(plots_dir, "training.pdf")
    if os.path.exists(csv_loss_fn) and not os.path.exists(plot_training_fn):
        csv_to_plot(csv_loss_fn, plot_training_fn, \
                "epoch", "loss", \
                os.path.basename(embedding.embedding_dir) + ": Training")
        plotted = True
    print("Done" if plotted else "Nothing to plot / Already plotted")

def analyse_embedding(embedding, output_dir, fn_synonyms_id, func_delta_r=None):
    # prepare output directory
    output_dir = output_dir.rstrip("/")
    prepare_output_dir(embedding, output_dir)
    evaluation_dir = os.path.join(output_dir, "evaluation")
    plots_dir = os.path.join(output_dir, "plots")
    
    # set default names
    rel_mat_similarity_name = "df_rel_mat_similarity_{0}"
    rel_nearest_neighbors_name = "df_rel_nearest_neighbors_{0}"
    nn_name = "nn_{0}"
    class_name = "class_{0:.2f}_percent_{1}"
    
    # get modelspecific func
    if not func_delta_r:
        if embedding.embedding_model == models.RESCAL:
            func_delta_r = get_rescal_delta_r
        elif embedding.embedding_model == models.TransE:
            func_delta_r = get_transe_delta_r
        elif embedding.embedding_model == models.TransH:
            func_delta_r = get_transh_delta_r
        elif embedding.embedding_model == models.TransR:
            func_delta_r = get_transr_delta_r
        elif embedding.embedding_model == models.TransD:
            func_delta_r = get_transd_delta_r
        elif embedding.embedding_model == models.DistMult:
            func_delta_r = get_distmult_delta_r
        elif embedding.embedding_model == models.HolE:
            func_delta_r = get_hole_delta_r
        elif embedding.embedding_model == models.ComplEx:
            func_delta_r = get_complex_delta_r
        elif embedding.embedding_model == models.Analogy:
            func_delta_r = get_analogy_delta_r
        else:
            # not implemented yet
            print("func_delta_r not implemented for specified model type.")
            return
    else:
        print("func_delta_r OVERRIDE")

    # SIMILARITY MATRIX L1: check if similarity matrix already exist
    fn_rel_mat_similarity_l1 = os.path.join(output_dir, rel_mat_similarity_name.format("l1.csv"))
    if os.path.exists(fn_rel_mat_similarity_l1):
        # load similarity matrix
        sys.stdout.write("Reading L1 Similarity Matrix ... ")
        sys.stdout.flush()
        df_rel_mat_similarity_l1 = pd.read_csv(fn_rel_mat_similarity_l1, index_col=0, encoding="utf-8")
        print("Done")
    else:
        # calculate similarity matrix
        df_rel_mat_similarity_l1 = calc_df_rel_mat_similarity(embedding, func_delta_r, norm_DELTA_r, 1)
        
        # save calculations as csv in output folder
        sys.stdout.write("Saving L1 Similarity Matrix ... ")
        sys.stdout.flush()
        df_rel_mat_similarity_l1.to_csv(fn_rel_mat_similarity_l1, index=True, encoding="utf-8")
        print("Done")
    
    # SIMILARITY MATRIX COS: check if similarity matrix already exist
    fn_rel_mat_similarity_cos = os.path.join(output_dir, rel_mat_similarity_name.format("cos.csv"))
    if os.path.exists(fn_rel_mat_similarity_cos):
        # load similarity matrix
        sys.stdout.write("Reading COS Similarity Matrix ... ")
        sys.stdout.flush()
        df_rel_mat_similarity_cos = pd.read_csv(fn_rel_mat_similarity_cos, index_col=0, encoding="utf-8")
        print("Done")
    else:
        # calculate similarity matrix
        df_rel_mat_similarity_cos = calc_df_rel_mat_similarity(embedding, func_delta_r, cos_DELTA_r, 2)
        
        # save calculations as csv in output folder
        sys.stdout.write("Saving COS Similarity Matrix ... ")
        sys.stdout.flush()
        df_rel_mat_similarity_cos.to_csv(fn_rel_mat_similarity_cos, index=True, encoding="utf-8")
        print("Done")
    
    # NEAREST NEIGHBORS L1: check if exists
    fn_rel_nearest_neighbors_l1 = os.path.join(output_dir, rel_nearest_neighbors_name.format("l1.csv"))
    if os.path.exists(fn_rel_nearest_neighbors_l1):
        # load
        sys.stdout.write("Reading L1 nearest neighbors dataframe ... ")
        sys.stdout.flush()
        df_rel_nearest_neighbors_l1 = pd.read_csv(fn_rel_nearest_neighbors_l1, index_col=0, encoding="utf-8")
        print("Done")
    else:
        # getting nearest neighbors dataframe
        df_rel_nearest_neighbors_l1 = get_df_rel_nearest_neighbors(embedding, df_rel_mat_similarity_l1, True)
        
        # save nearest neighbors dataframe
        sys.stdout.write("Saving L1 nearest neighbors dataframe ... ")
        sys.stdout.flush()
        df_rel_nearest_neighbors_l1.to_csv(fn_rel_nearest_neighbors_l1, index=True, encoding="utf-8")
        print("Done")

    # NEAREST NEIGHBORS COS: check if exists
    fn_rel_nearest_neighbors_cos = os.path.join(output_dir, rel_nearest_neighbors_name.format("cos.csv"))
    if os.path.exists(fn_rel_nearest_neighbors_cos):
        # load
        sys.stdout.write("Reading COS nearest neighbors dataframe ... ")
        sys.stdout.flush()
        df_rel_nearest_neighbors_cos = pd.read_csv(fn_rel_nearest_neighbors_cos, index_col=0, encoding="utf-8")
        print("Done")
    else:
        # getting nearest neighbors dataframe
        df_rel_nearest_neighbors_cos = get_df_rel_nearest_neighbors(embedding, df_rel_mat_similarity_cos, False)
        
        # save nearest neighbors dataframe
        sys.stdout.write("Saving COS nearest neighbors dataframe ... ")
        sys.stdout.flush()
        df_rel_nearest_neighbors_cos.to_csv(fn_rel_nearest_neighbors_cos, index=True, encoding="utf-8")
        print("Done")

    # DETECT, EVALUATE AND SAVE SYNONYMS
    strinfo = "\rDetecting, evaluating and saving Synonyms ... "
    sys.stdout.write(strinfo)
    sys.stdout.flush()

    # set flag indicating wether we can evaluate or not by checking
    # if synonyms list was specified
    evaluation = fn_synonyms_id and os.path.exists(fn_synonyms_id)

    # prepare evaluation
    if evaluation:
        evaluation_nn_l1 = []
        evaluation_nn_cos = []
        evaluation_class_l1 = []
        evaluation_class_cos = []
        synonyms = get_synonyms_set(fn_synonyms_id)
        def precision_recall(l):
            precision, recall = calc_precision_recall(l, synonyms)
            return [precision, recall]
        def save_evaluation(evaluation, file_name):
            evaluation_file = open(file_name, "w")
            evaluation_file.writelines(map(lambda x: str(x[0]) + "\t" + str(x[1]) + "\n", evaluation))
            evaluation_file.close()
        def plot_evaluation(evaluation, file_name, xlabel, ylabel, title):
            # get 2-tuple values from evaluation
            get_val1 = lambda x: list(map(lambda y: float(y[0]), x))
            get_val2 = lambda x: list(map(lambda y: float(y[1]), x))
            get_valid_num = lambda x: len(list(filter(lambda y: y > 0.0, x)))
            val1 = get_val1(evaluation)
            val2 = get_val2(evaluation)
            min_idx = len(val1) - min(get_valid_num(val1), get_valid_num(val2))
            max_val_x = max(val2)
            max_val_y = max(val1)
            max_val_x += max_val_x / 10.0
            max_val_y += max_val_y / 10.0
            # plot 2-tuple values curve and save it
            plt.clf()
            plt.xlim([0, max_val_x if max_val_x > 0.0 and max_val_x < 1.0 else 1])
            plt.ylim([0, max_val_y if max_val_y > 0.0 and max_val_y < 1.0 else 1])
            plt.plot(val2[min_idx:], val1[min_idx:])
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.savefig(file_name)

    # nearest neighbors
    nn_l1 = nearest_neighbors(df_rel_mat_similarity_l1, True)
    nn_cos = nearest_neighbors(df_rel_mat_similarity_cos, False)
    save_synonyms_set(nn_l1, os.path.join(evaluation_dir, nn_name.format("l1")), embedding)
    save_synonyms_set(nn_cos, os.path.join(evaluation_dir, nn_name.format("cos")), embedding)
    if evaluation:
        evaluation_nn_l1.append(precision_recall(nn_l1))
        evaluation_nn_cos.append(precision_recall(nn_cos))
        save_evaluation(evaluation_nn_l1, os.path.join(evaluation_dir, "evaluation_nn_l1.txt"))
        save_evaluation(evaluation_nn_cos, os.path.join(evaluation_dir, "evaluation_nn_cos.txt"))

    # left threshold list (0.1% - 5%, += 0.1%)
    max_population_percents_left = list(np.linspace(0.1, 5, 50))
    # mid threshold list (5% - 10%, += 0.2%)
    max_population_percents_mid = list(np.linspace(5, 20, 76))
    # right threshold list (10% - 100%, += 1%)
    max_population_percents_right = list(np.linspace(20, 100, 81))
    # define our dbpedia threshold list (0.1% - 3%, += 0.1%)
    # (to reduce manual evaluation overhead)
    max_population_percents_dbpedia = list(np.linspace(0.1, 5, 50))
    # combine
    max_population_percents = max_population_percents_left[:-1] \
            + max_population_percents_mid[:-1] \
            + max_population_percents_right
    #max_population_percents = max_population_percents_dbpedia   # ONLY FOR DBPEDIA EVALUATION
    
    # do for every percent for max_population_percent
    max_iteration = len(max_population_percents)
    iteration = 0
    for max_population_percent in max_population_percents:
        # calculate synonyms
        class_l1 = classification(df_rel_mat_similarity_l1, max_population_percent * 0.01, True)
        class_cos = classification(df_rel_mat_similarity_cos, max_population_percent * 0.01, False)
        # save synonyms
        save_synonyms_set(class_l1, os.path.join(evaluation_dir, \
                class_name.format(max_population_percent, "l1")), embedding)
        save_synonyms_set(class_cos, os.path.join(evaluation_dir, \
                class_name.format(max_population_percent, "cos")), embedding)
        # calculate precision and recall
        if evaluation:
            evaluation_class_l1.append(precision_recall(class_l1))
            evaluation_class_cos.append(precision_recall(class_cos))
        # output strinfo
        percentage = int(float(iteration * 100) / max_iteration)
        sys.stdout.write(strinfo + "%s%%" % percentage)
        sys.stdout.flush()
        iteration = iteration + 1
    # save evaluation with plots for classifications
    if evaluation:
        save_evaluation(evaluation_class_l1, os.path.join(evaluation_dir, "evaluation_class_l1.txt"))
        save_evaluation(evaluation_class_cos, os.path.join(evaluation_dir, "evaluation_class_cos.txt"))
        plot_evaluation(evaluation_class_l1, \
                os.path.join(plots_dir, "evaluation_class_l1.pdf"), \
                "recall", "precision", \
                os.path.basename(embedding.embedding_dir) + ": Evaluation (L1-norm distance)")
        plot_evaluation(evaluation_class_cos, \
                os.path.join(plots_dir, "evaluation_class_cos.pdf"), \
                "recall", "precision", \
                os.path.basename(embedding.embedding_dir) + ": Evaluation (cosine similarity)")
    # done
    print(strinfo + "Done")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MAIN METHOD FOR AUTOMATED ANALYSIS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def main():
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("MODEL_TYPE", type=str, choices=Embedding.MODELS.keys(), \
            help="Model type of INPUT_EMBEDDING")
    parser.add_argument("INPUT_BENCHMARK", type=str, \
            help="Benchmark directory of INPUT")
    parser.add_argument("INPUT_EMBEDDING", type=str, \
            help="Embedding directory of INPUT")
    parser.add_argument("OUTPUT_DIR", type=str, \
            help="Directory where to output the whole synonym analysis")
    parser.add_argument("-d", "--dimension", type=int, default=100, \
            help="Hidden dimensions of INPUT_EMBEDDING (Default: 100)")
    parser.add_argument("-g", "--ground-truth-available", action="store_true", default=False, \
            help="An optional boolean flag indicating wether INPUT_BENCHMARK contains ground-truth" \
            + " synonyms file 'synonyms_id.txt' which should be evaluated (Default: False).")
    parser.add_argument("-o", "--override-delta", type=str, choices=Embedding.MODELS.keys(), \
            required=False, default=None, \
            help="An optional func_delta_r override (MUST BE COMPATIBLE!) (Default: None).")
    args = parser.parse_args()

    # check delta override
    func_delta_r = None
    if args.override_delta:
        if args.override_delta == "rescal":
            func_delta_r = get_rescal_delta_r
        elif args.override_delta == "transe":
            func_delta_r = get_transe_delta_r
        elif args.override_delta == "transh":
            func_delta_r = get_transh_delta_r
        elif args.override_delta == "transr":
            func_delta_r = get_transr_delta_r
        elif args.override_delta == "transd":
            func_delta_r = get_transd_delta_r
        elif args.override_delta == "distmult":
            func_delta_r = get_distmult_delta_r
        elif args.override_delta == "hole":
            func_delta_r = get_hole_delta_r
        elif args.override_delta == "complex":
            func_delta_r = get_complex_delta_r
        elif args.override_delta == "analogy":
            func_delta_r = get_analogy_delta_r
        else:
            # not implemented
            print("func_delta_r not implemented for model " + args.override_delta + ".")
            sys.exit(1)

    # import embedding
    embedding = Embedding(args.INPUT_BENCHMARK, \
            args.INPUT_EMBEDDING, \
            Embedding.MODELS[args.MODEL_TYPE], \
            args.dimension)

    # analyse embedding
    fn_synonyms_id = os.path.join(args.INPUT_BENCHMARK, "synonyms_id.txt") \
            if args.ground_truth_available else None
    analyse_embedding(embedding, args.OUTPUT_DIR, fn_synonyms_id, func_delta_r)

if __name__ == "__main__":
    main()

