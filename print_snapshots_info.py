import numpy as np
import os
import argparse
import logging
from joblib import Parallel, delayed
from tqdm import tqdm

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.time_graph_generators as generators
from anonygraph.constants import *

logger = logging.getLogger(__file__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_sequence_data_argument(parser)
    rutils.add_log_argument(parser)
    rutils.add_workers_argument(parser)

def calculate_added_edges(entity_id, pre_subgraph, subgraph):
    result = 0

    if subgraph.is_entity_id(entity_id):
        for node1_id, relation_id, node2_id in subgraph.get_out_edges_iter(entity_id):
            if not pre_subgraph.is_edge_existed(node1_id, relation_id, node2_id):
                result += 1

        for node1_id, relation_id, node2_id in subgraph.get_in_edges_iter(entity_id):
            if not pre_subgraph.is_edge_existed(node1_id, relation_id, node2_id):
                result += 1

    return result

def calculate_removed_edges(entity_id, pre_subgraph, subgraph):
    result = 0

    if pre_subgraph.is_entity_id(entity_id):
        for node1_id, relation_id, node2_id in pre_subgraph.get_out_edges_iter(entity_id):
            if not subgraph.is_edge_existed(node1_id, relation_id, node2_id):
                result += 1

        for node1_id, relation_id, node2_id in pre_subgraph.get_in_edges_iter(entity_id):
            if not subgraph.is_edge_existed(node1_id, relation_id, node2_id):
                result += 1

    return result

def calculate_num_changes_edges(data):
    # logger.info(data)
    data_name, sample, strategy, entity_id, pre_t, t, args = data

    if pre_t == -1:
        return 0

    pre_subgraph = dutils.load_raw_subgraph(data_name, sample, strategy, pre_t, args)
    subgraph = dutils.load_raw_subgraph(data_name, sample, strategy, t, args)

    num_added_edges = calculate_added_edges(entity_id, pre_subgraph, subgraph)
    num_removed_edges = calculate_removed_edges(entity_id, pre_subgraph, subgraph)

    return num_added_edges + num_removed_edges

def calculate_average_changes_edges(data_name, sample, strategy, time_instances, args):
    # get all users
    entity_ids = dutils.get_raw_entity_indexes(data_name, sample).values()

    subgraph_data = []
    for entity_id in entity_ids:
        pre_t = -1

        for t in time_instances:
            subgraph_data.append((data_name, sample, strategy, entity_id, pre_t, t, args))
            pre_t = t

    logger.debug("subgraph data: {}".format(subgraph_data[0:10]))
    results = list(
        Parallel(n_jobs=args["workers"])(
            delayed(calculate_num_changes_edges)(subgraph_item) for subgraph_item in tqdm(subgraph_data)
    ))
    logger.debug("number of results: {}".format(len(results)))
    logger.debug("results: {}".format(results))

    return np.mean(results)


def get_time_insts(data_name, sample, strategy, args):
    raw_sequence_path = putils.get_sequence_raw_subgraphs_path(data_name, sample, strategy, args)
    folders = os.listdir(raw_sequence_path)
    time_instances = []
    for folder in folders:
        path = os.path.join(raw_sequence_path, folder)
        if os.path.isdir(path):
            time_instances.append(int(folder))

    return sorted(time_instances)

def main(args):
    logger.info(args)
    data_name = args["data"]
    sample = args["sample"]
    strategy = args["strategy"]

    time_instances = get_time_insts(data_name, sample, strategy, args)

    logger.debug("time instances: {}".format(time_instances))

    average_changed_edges = calculate_average_changes_edges(data_name, sample, strategy, time_instances, args)
    logger.info("average changed eddges: {}".format(average_changed_edges))

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
