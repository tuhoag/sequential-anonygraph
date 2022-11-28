import numpy as np
import glob
from tkinter import N
from tqdm import tqdm
from time import time
from joblib import Parallel, delayed
import os
import logging
import math

import pandas as pd
import json
import glob
from heapq import nlargest

import anonygraph.utils.path as putils
import anonygraph.evaluation.clusters_metrics as cmetrics
import anonygraph.evaluation.subgraphs_metrics as gmetrics
import anonygraph.algorithms as algo
import anonygraph.utils.data as dutils
from anonygraph.constants import *

logger = logging.getLogger(__name__)

def get_exp_data(
    exp_path, prepare_data_fn, prepare_data_args, workers, refresh, args
):
    if refresh or not os.path.exists(exp_path):
        logger.info('preparing data')
        raw_data = prepare_data_fn(prepare_data_args, workers, args)

        if not os.path.exists(os.path.dirname(exp_path)):
            logger.info('creating folder: {}'.format(os.path.dirname(exp_path)))
            os.makedirs(os.path.dirname(exp_path))

        logger.debug(raw_data)
        df = pd.DataFrame(raw_data)
        logger.info('saving data to: {}'.format(exp_path))
        df.to_csv(exp_path, index=False)
    else:
        logger.info('reading data from: {}'.format(exp_path))
        df = pd.read_csv(exp_path)

    return df

def get_cluster_quality(clusters, subgraph, args):
    metrics_names = cmetrics.get_all_metric_names()

    return cmetrics.calculate_quality_metrics(
        metrics_names, clusters, subgraph, args
    )


def get_clusters_quality_from_path(clusters_path):
    clusters = algo.Clusters.from_file(clusters_path)
    info = putils.extract_info_from_clusters_path(clusters_path)

    subgraph = dutils.load_raw_subgraph(
        info["data"], info["sample"], info["strategy"], info["t"], info
    )

    quality_info = get_cluster_quality(clusters, subgraph, info)
    info.update(quality_info)

    return info

def prepare_clusters_data(data_info, num_workers, args):
    data_name, sample, strategy, anony_mode = data_info["data"], data_info[
        "sample"], data_info["strategy"], data_info["anony_mode"]

    clusters_dir_path = putils.get_all_clusters_path(
        data_name, sample, strategy, anony_mode, args
    )
    logger.debug("clusters path: {}".format(clusters_dir_path))

    clusters_paths = glob.glob(clusters_dir_path + "/*")
    logger.info("preparing data from {} clusters".format(len(clusters_paths)))
    logger.debug(clusters_paths)

    start_time = time()
    raw_data = list(
        Parallel(n_jobs=num_workers)(
            delayed(get_clusters_quality_from_path)(path)
            for path in tqdm(clusters_paths)
        )
    )
    logger.debug(raw_data)
    logger.info(
        "finished preparing data in {} seconds".format(time() - start_time)
    )

    return raw_data

def get_anonymized_graph_quality(anonymized_graph, graph, args):
    metrics_names = gmetrics.get_all_metric_names()

    return gmetrics.calculate_quality_metrics(
        metrics_names, anonymized_graph, graph, args
    )


def get_subgraphs_quality_from_path(graph_path):
    # subgraph = data.SubGraph.from_index_and_edges_data
    info = putils.extract_info_from_anonymized_subgraph_path(graph_path)
    logger.debug(info)

    anonymized_subgraph = dutils.get_anonymized_subgraph(
        data_name=info["data"],
        sample=info["sample"],
        strategy=info["strategy"],
        time_instance=info["t"],
        info_loss_name=info["info_loss"],
        k=info["k"],
        w=info["w"],
        reset_w=info["reset_w"],
        l=info["l"],
        enforcer_name=info["enforcer"],
        calgo=info["calgo"],
        galgo=info["galgo"],
        anony_mode=info["anony_mode"],
        args=info
    )

    raw_subgraph = dutils.load_raw_subgraph(
        info["data"], info["sample"], info["strategy"], info["t"], info
    )

    quality_info = get_anonymized_graph_quality(
        anonymized_subgraph, raw_subgraph, info
    )
    info.update(quality_info)

    return info


def prepare_anonymized_subgraphs_data(data_info, num_workers, args):
    data_name, sample, strategy, anony_mode = data_info["data"], data_info[
        "sample"], data_info["strategy"], data_info["anony_mode"]

    graphs_dir_path = putils.get_all_anonymized_subgraphs_dir(
        data_name, sample, strategy, anony_mode, args
    )
    logger.debug("graphs path: {}".format(graphs_dir_path))

    graph_paths = glob.glob(graphs_dir_path + "/*")
    logger.info("preparing data from {} graphs".format(len(graph_paths)))
    logger.debug(graph_paths)

    start_time = time()
    raw_data = list(
        Parallel(n_jobs=num_workers)(
            delayed(get_subgraphs_quality_from_path)(path)
            for path in tqdm(graph_paths)
        )
    )
    logger.debug(raw_data)
    logger.info(
        "finished preparing data in {} seconds".format(time() - start_time)
    )

    return raw_data


def get_title(name):
    name2title = {
        "adm": "Average Information Loss of All Users",
        "radm": "Average Information Loss of Anonymized Users",
        "ratio_intersection_entities": "Intersection Entities (%)",
        "ratio_intersection_edges": "Intersection Edges (%)",
        "ratio_fake_entities": "Ratio of Fake Entities (%)",
        ANONYMIZED_ANONYMITY_METRIC: "Anonymity",
        "ratio_big_clusters": "Ratio of Big Clusters (%)",
        "ratio_entities_in_big_clusters": "Ratio of Entities in Big Clusters (%)",
        "calgo": "Clustering Algorithms",
        "k": "k",
        "l": "l",
        "reset_w": "Reset Window Size",
        "reset_w_name": "Reset Window Size",
        "max_dist": r"$\tau$",
        "calgo_enforcer": "Algorithm",
        "ratio_fake_removed_edges": "Ratio of Fake/Removed Edges (%)",
        "ratio_fake_removed_entities": "Ratio of Fake/Removed Users (%)",
        "algo": "Algorithm",
        # "clustering_and_gen": ""
    }

    return name2title[name]

def get_xticks(name, values, num_sticks=-1):
    if name == "t":
        if num_sticks == -1:
            x_ticks = values
        else:
            sorted_values = sorted(values)
            step_value = math.floor(len(values) / num_sticks)
            if step_value == 0:
                step_value = 1
            logger.debug("step_value: {}".format(step_value))
            x_ticks = []
            for val_idx in range(0, len(sorted_values), step_value):
                # logger.debug("val_idx: {}".format(val_idx))
                x_ticks.append(sorted_values[val_idx])

            # x_ticks.append(sorted_values[-1])
            logger.debug("xticks: {}".format(x_ticks))
    elif name in ["k", "max_dist", "l"]:
        x_ticks = values
    else:
        raise Exception("Unsupported get_xticks of {}".format(name))

    return x_ticks


def load_training_data(args, suffix=""):
    data_name = args["data"]
    strategy_name = putils.get_strategy_name(args["strategy"], args)
    path = os.path.join("exp_data", "training", "{}_{}{}.json".format(data_name, strategy_name,suffix))

    if os.path.exists(path):
        dataf = open(path, "r")

        try:
            data = json.load(dataf)
            # print(type(data))
            dataf.close()
        except ValueError as e:
            data = {}
    else:
        data = {}
    # print(type(data))
    # raise Exception()
    return data

# def prepare_clusters_data(data_info, num_workers, args):
def prepare_training_data(data_info, num_workers, args):
    data_name, sample, strategy = data_info["data"], data_info["sample"], data_info["strategy"]
    dir_path = putils.get_training_result_file_dir()
    data_str = putils.get_raw_graph_dir_name(data_name, sample)
    strategy_str = putils.get_strategy_name(strategy, args)

    model_paths = glob.glob(os.path.join(dir_path, "{}_{}_*".format(data_str, strategy_str)))

    logger.debug(model_paths)
    all_results = []
    for model_path in model_paths:
        all_raw_anony_paths = glob.glob(os.path.join(model_path,"*"))

        logger.debug(all_raw_anony_paths)
        for raw_anony_path in all_raw_anony_paths:
            all_ts_paths = glob.glob(os.path.join(raw_anony_path,"*"))
            logger.debug(all_ts_paths)
            for ts_path in all_ts_paths:
                results = load_model_training_data(ts_path)
                all_results.append(results)
        # for all_ts_path in all_raw_anony_paths:
        #     for ts_path in all_ts_path:
        #         logger.debug(ts_path)
        # load_model_training_data(model_path)

    # logger.debug(results)
                # raise Exception()

    return all_results

def extract_training_path_properties(path):
    info = {}

    data = path.split(os.sep)
    logger.debug(data)
    data_name, sample, sattr, strategy_name, n_sg, d = data[2].split("_")

    setting_name = data[3]
    if setting_name != "raw":
        k, l, reset_w, calgo, enforcer_str = data[3].split("_")
        enforcer_name,max_dist = enforcer_str.split("#")


    else:
        k = 1
        l = 1
        reset_w = -1
        calgo = "raw"
        enforcer_name = "raw"
        max_dist = 1.0
        enforcer_str = "raw"


    t = data[4].split(".json")[0]

    info = {
        "data": data_name,
        "sample": int(sample),
        "strategy": strategy_name,
        "n_sg": int(n_sg),
        "d": int(d),
        "setting_name": setting_name,
        "t": int(t),
        "k": int(k),
        "l": int(l),
        "reset_w": int(reset_w),
        "calgo": calgo,
        "enforcer_str": enforcer_str,
        "sattr":sattr,
        "max_dist": float(max_dist),
        "enforcer_name": enforcer_name,
    }

    return info

def load_model_training_data(path):
    # extract path properties
    info = extract_training_path_properties(path)

    logger.debug(path)
    logger.debug(info)
    with open(path, "r") as f:
        data = json.load(f)
        metric_names = data.keys()

        for metric_name in metric_names:
            info[metric_name] = np.mean(data[metric_name]["best_results"])
            logger.debug(data[metric_name]["best_results"])

            # top
            top_list = [1, 3, 5, 10, 50]

            for top in top_list:
                values = nlargest(top, data[metric_name]["best_results"])
                logger.debug(values)

                # raise Exception()
                key ="top-{}_{}".format(top, metric_name)
                info[key] = np.mean(values)
        # logger.debug(data.keys())
    # file_name = os.path.basename(path)
    # logger.debug("{}:{}".format(path, file_name))
    # read file data
    # pass

    return info

def write_training_data(data, args):
    logger.debug(data)
    data_name = args["data"]
    strategy_name = putils.get_strategy_name(args["strategy"], args)
    path = os.path.join("exp_data", "training", "{}_{}.json".format(data_name, strategy_name))
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(path, 'w', encoding='utf-8') as f:
    # with open(path, "w") as f:
        json_str = json.dumps(data)
        json.dump(data, f, ensure_ascii=False, indent=4)
        # json.dump(json_str, f)

def write_snapshot_training_data(path, data, args):
    dir_name = os.path.dirname(path)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

