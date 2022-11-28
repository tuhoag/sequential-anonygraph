from time import time
from tqdm import tqdm
from joblib import Parallel, delayed
import glob
import argparse
import logging
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.utils.visualization as visual
import anonygraph.time_graph_generators as generators
import anonygraph.algorithms.clustering as calgo
import anonygraph.algorithms as algo
import anonygraph.evaluation.clusters_metrics as cmetrics
import anonygraph.constants as constants

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)

    parser.add_argument("--refresh", type=rutils.str2bool)

def calculate_num_edges(graph, t):
    return graph.get_num_edges(t)

def calculate_num_entities(graph, t):
    return graph.get_num_entities(t)

def get_previous_time_instance(graph, t):
    time_instances = sorted(graph.time_instances)
    t_index = time_instances.index(t)
    if t_index > 0:
        pre_t = time_instances[t_index-1]
    else:
        pre_t = t

    return pre_t

def calculate_num_new_edges(graph, t):
    pre_t = get_previous_time_instance(graph, t)

    count = 0
    for node1_id, relation_id, node2_id in graph.get_edges_iter(t):
        if not graph.has_edge_id(node1_id, relation_id, node2_id, pre_t):
            count += 1
    return count

def calculate_num_new_entities(graph, t):
    pre_t = get_previous_time_instance(graph, t)

    count = 0
    for node_id in graph.get_edges_iter(t):
        if not graph.has_node_id(node_id, pre_t):
            count += 1

    return count

def get_dynamic_graph_info(graph):
    time_instances = sorted(graph.time_instances)

    raw_data = []

    for t in tqdm(time_instances):
        info = {
            "t": t,
            "num_edges": calculate_num_edges(graph, t),
            "num_entities": calculate_num_entities(graph, t),
            "num_new_edges": calculate_num_new_edges(graph, t),
            "num_new_entities": calculate_num_new_entities(graph, t),
        }

        raw_data.append(info)

    return raw_data

def prepare_data(data_info, num_workers, args):
    data_name = data_info["data"]
    sample = data_info["sample"]

    # load dynamic graph
    graph = dutils.load_dynamic_graph_from_output_file(args["data"], args["sample"])

    logger.info("load dynamic graph: {}".format(graph))

    raw_data = get_dynamic_graph_info(graph)

    logger.debug(raw_data)

    return raw_data


def add_more_info(df):
    df["ratio_fake_edges"] = df[constants.FAKE_EDGES_METRIC
                               ] / df[constants.REAL_EDGES_METRIC]
    df["ratio_fake_entities"] = df[constants.FAKE_ENTITIES_METRIC
                                  ] / df[constants.REAL_ENTITIES_METRIC]
    df[r"$\tau$"] = df["max_dist"]


def visualize_fine_tune(
    df
):
    logger.debug(df)
    df.sort_values(by=["t"], inplace=True)

    logger.info("visualizing data: {}".format(df))
    t_values = df["t"].unique()

    logger.debug("t_values: {}".format(t_values))



def main(args):
    logger.debug(args)
    data_path = putils.get_dynamic_graph_exp_data_path(
        args["data"], args["sample"]
    )

    df = visual.get_exp_data(
        exp_path=data_path,
        prepare_data_fn=prepare_data,
        prepare_data_args={
            "data": args["data"],
            "sample": args["sample"],
        },
        workers=args["workers"],
        refresh=args["refresh"],
        args=args
    )

    # add_more_info(df)

    logger.debug(df)
    logger.info("visualizing")
    visualize_fine_tune(
        df=df
    )


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
