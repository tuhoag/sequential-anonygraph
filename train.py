import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import DGLDataset
from anonygraph.evaluation.classification.data_loader import DummyGraph, AnonyGraph
from anonygraph.evaluation.classification.rgcn_model import RGCN
from anonygraph.evaluation.classification.trainer import train
import anonygraph.utils.path as putils
import anonygraph.utils.runner as rutils
import anonygraph.utils.visualization as vutils
import warnings
import json
import os
import itertools

logger = logging.getLogger(__name__)

warnings.filterwarnings(action='ignore', category=UserWarning)

def add_arguments(parser):
    # rutils.add_data_argument(parser)
    # rutils.add_sequence_data_argument(parser)
    rutils.add_graph_generalization_argument(parser)

    parser.add_argument('--d_list', type=rutils.string2list(int))
    parser.add_argument('--k_list', type=rutils.string2list(int))
    parser.add_argument('--w_list', type=rutils.string2list(int))
    parser.add_argument('--max_dist_list', type=rutils.string2list(float))
    parser.add_argument('--l_list', type=rutils.string2list(int))
    parser.add_argument("--calgo_list", type=rutils.string2list(str))
    parser.add_argument("--reset_w_list", type=rutils.string2list(int))
    parser.add_argument("--update", type=rutils.str2bool)

    parser.add_argument("--anony", type=rutils.str2bool, default="y")

    # rutils.add_workers_argument(parser)

    rutils.add_log_argument(parser)
    # parser.add_argument("--refresh", type=rutils.str2bool)


def train_raw_data(t, d, args):
    raw_subgraph_path = putils.get_raw_subgraph_path(
        args["data"], args["sample"], args["strategy"], t, args
    )

    # key = putils.get_anonymized_subgraph_name(t, args["info_loss"], k, w, l, reset_w, calgo, enforcer, args["galgo"], args)

    # path = putils.get_anonymized_subgraph_path("yago15", -1, "mean", 0, , "all", args)
    logger.debug(raw_subgraph_path)

    # return
    graph = AnonyGraph()
    graph.load(raw_subgraph_path)

    if graph.graph is None:
        return None

    logger.debug(graph.graph)
    logger.debug(graph.graph.ntypes)

    logger.debug(graph.graph.etypes)

    logger.debug(graph.graph.canonical_etypes)

    logger.debug(graph.graph.nodes("user"))

    # logger.debug(graph.value2value_idx)
    # logger.debug(graph.label_id2idxes)
    n_features = len(graph.value2value_idx)
    n_labels = len(graph.label_id2idxes)
    model = RGCN(n_features, d, n_labels, graph.graph.etypes)

    # return
    results = train(model, graph.graph, 10000, 10, 0.0001)
    return results

def train_anonymized_data(t, k, w, l, reset_w, calgo, enforcer, args):
    anony_subgraph_path = putils.get_anonymized_subgraph_path(
        args["data"], args["sample"], args["strategy"], t,
        args["info_loss"], k, w, l, reset_w, calgo, enforcer, args["galgo"], args["anony_mode"],
        args
    )

    if not os.path.exists(anony_subgraph_path):
        logger.debug("skip: graph do not exist")
        return None

    # key = putils.get_anonymized_subgraph_name(t, args["info_loss"], k, w, l, reset_w, calgo, enforcer, args["galgo"], args)

    # path = putils.get_anonymized_subgraph_path("yago15", -1, "mean", 0, , "all", args)
    logger.debug(anony_subgraph_path)

    # return
    graph = AnonyGraph()
    graph.load(anony_subgraph_path)

    if graph.graph is None:
        logger.debug("skip: graph cannot be loaded")
        return None

    logger.debug(graph.graph)
    logger.debug(graph.graph.ntypes)

    logger.debug(graph.graph.etypes)

    logger.debug(graph.graph.canonical_etypes)

    # logger.debug(graph.graph.nodes("user"))

    logger.debug(graph.value2value_idx)
    # logger.debug(graph.label_id2idxes)
    n_features = len(graph.value2value_idx)
    logger.debug("n_features:{}".format(n_features))
    n_labels = len(graph.label_id2idxes)
    model = RGCN(n_features, 500, n_labels, graph.graph.etypes)

    # return
    results = train(model, graph.graph, 10000, 10, 0.0001)
    return results

def convert_results_to_list(train_results):
    results = []
    for result in train_results:
        results.append(result.tolist())

    return results

def get_training_raw_data(data, d_list, args):
    update = args["update"]
    strategy_name = args["strategy"]

    if strategy_name == "mean":
        n_sg = args["n_sg"]
    else:
        n_sg = 1

    for d in d_list:
        model_data = data.get(str(d), None)
        if model_data is None:
            model_data = {}
            data[d] = model_data

        # raw data
        # logger.debug(data.keys())
        raw_data = model_data.get("raw", None)
        # logger.debug(model_data.keys())
        if raw_data is None:
            raw_data = {}
            model_data["raw"] = raw_data

        for t in range(n_sg):
            logger.info("raw_{}".format(t))

            t_model_data = raw_data.get(str(t), None)
            # logger.debug(raw_data.keys())
            # # logger.debug(t_model_data)
            # logger.debug(update)
            # logger.debug(t_model_data is not None)
            # logger.debug(not update)

            if t_model_data is not None and not update:
                continue

            if t_model_data is None:
                t_model_data = {}


            # logger.debug(t)
            results = train_raw_data(t, d, args)

            if results is None:
                continue

            for metric_name, metric in results.items():
                t_model_data[metric_name] = {
                    "best_results": convert_results_to_list(metric.best_result),
                    "all_results": convert_results_to_list(metric.all_results),
                    "epoches": metric.epoches,
                }

            raw_data[t] = t_model_data
            vutils.write_training_data(data, args)


def get_training_raw_data2(d_list, args):
    update = args["update"]
    strategy_name = args["strategy"]

    if strategy_name == "mean":
        n_sg = args["n_sg"]
    else:
        n_sg = 1

    for d in d_list:
        for t in range(n_sg):
            logger.info("raw_{}".format(t))

            # check if file exist
            path = putils.get_raw_graph_training_result_file_path(args["data"], strategy_name, d, t, args)

            logger.debug(not os.path.exists(path))
            logger.debug(not update)
            if os.path.exists(path) and not update:
                continue


            t_model_data = {}


            # logger.debug(t)
            results = train_raw_data(t, d, args)

            if results is None:
                continue

            for metric_name, metric in results.items():
                t_model_data[metric_name] = {
                    "best_results": convert_results_to_list(metric.best_result),
                    "all_results": convert_results_to_list(metric.all_results),
                    "epoches": metric.epoches,
                }


            vutils.write_snapshot_training_data(path, t_model_data, args)

def get_training_anony_data2(d_list, args):
    update = args["update"]
    strategy_name = args["strategy"]
    k_list = args["k_list"]
    l_list = args["l_list"]
    reset_w_list = args["reset_w_list"]
    calgo_list = args["calgo_list"]
    max_dist_list = args["max_dist_list"]

    if strategy_name == "mean":
        n_sg = args["n_sg"]
    else:
        n_sg = 1

    for d in d_list:
        for k, l,reset_w, calgo, max_dist in itertools.product(k_list,l_list,reset_w_list,calgo_list, max_dist_list):
            current_args = args.copy()
            current_args.update({
                "k": k,
                "l": l,
                "reset_w": reset_w,
                "calgo": calgo,
                "max_dist": max_dist,
            })
            enforcer_str = putils.get_enforcer_str(current_args["enforcer"], current_args)
            key = "{}_{}_{}_{}_{}".format(k,l,reset_w,calgo,enforcer_str)

            for t in range(n_sg):
                logger.info("{}_{}".format(key, t))
                path = putils.get_anony_graph_training_result_file_path(args["data"], strategy_name, d, k,l,reset_w,calgo,enforcer_str, t, args)

                logger.debug(os.path.exists(path))
                logger.debug(not update)

                if os.path.exists(path) and not update:
                    logger.debug("skip")
                    continue
                t_model_data = {}

                results = train_anonymized_data(t, k, -1, l, reset_w, calgo, current_args["enforcer"], current_args)

                logger.debug("result: {}".format(results))
                if results is None:
                    continue

                for metric_name, metric in results.items():
                    t_model_data[metric_name] = {
                        "best_results": convert_results_to_list(metric.best_result),
                        "all_results": convert_results_to_list(metric.all_results),
                        "epoches": metric.epoches,
                    }

                vutils.write_snapshot_training_data(path, t_model_data, args)

def get_training_anony_data(data, d_list, args):
    update = args["update"]
    strategy_name = args["strategy"]
    k_list = args["k_list"]
    l_list = args["l_list"]
    reset_w_list = args["reset_w_list"]
    calgo_list = args["calgo_list"]
    max_dist_list = args["max_dist_list"]

    if strategy_name == "mean":
        n_sg = args["n_sg"]
    else:
        n_sg = 1

    for d in d_list:
        model_data = data.get(str(d), None)
        if model_data is None:
            model_data = {}
            data[d] = model_data

        for k, l,reset_w, calgo, max_dist in itertools.product(k_list,l_list,reset_w_list,calgo_list, max_dist_list):
            current_args = args.copy()
            current_args.update({
                "k": k,
                "l": l,
                "reset_w": reset_w,
                "calgo": calgo,
                "max_dist": max_dist,
            })
            enforcer_str = putils.get_enforcer_str(current_args["enforcer"], current_args)
            key = "{}_{}_{}_{}_{}".format(k,l,reset_w,calgo,enforcer_str)

            anony_model_data = model_data.get(key, None)
            if anony_model_data is None:
                anony_model_data = {}
                model_data[key] = anony_model_data

            for t in range(n_sg):
                logger.info("{}_{}".format(key, t))
                t_model_data = anony_model_data.get(str(t), None)

                # logger.debug(anony_model_data.keys())
                # # logger.debug(t_model_data)
                # logger.debug(update)
                # logger.debug(t_model_data is not None)
                # logger.debug(not update)
                # logger.debug(t_model_data is not None and not update)
                # raise Exception()
                if t_model_data is not None and not update:
                    continue

                if t_model_data is None:
                    t_model_data = {}

                results = train_anonymized_data(t, k, -1, l, reset_w, calgo, current_args["enforcer"], current_args)

                if results is None:
                    continue

                for metric_name, metric in results.items():
                    t_model_data[metric_name] = {
                        "best_results": convert_results_to_list(metric.best_result),
                        "all_results": convert_results_to_list(metric.all_results),
                        "epoches": metric.epoches,
                    }

                anony_model_data[t] = t_model_data
                vutils.write_training_data(data, current_args)

def main(args):
    # graph = DummyGraph()
    data_name = args["data"]
    strategy_name = args["strategy"]
    # strategy_name = putils.get_strategy_name(args["strategy"], args)


    # data = vutils.load_training_data(args)

    d_list = args["d_list"]
    k_list = args["k_list"]
    l_list = args["l_list"]
    reset_w_list = args["reset_w_list"]
    calgo_list = args["calgo_list"]
    max_dist_list = args["max_dist_list"]

    if args["anony"]:
        get_training_anony_data2(d_list, args)
    else:
        get_training_raw_data2(d_list, args)



def visualize(results):
    for metric_name, metric in results.items():
        plt.plot(metric.epoches, metric.all_results, label=metric_name)
    plt.legend()
    plt.savefig("test.pdf")
    plt.savefig("test.png")
    # plt.show()

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
