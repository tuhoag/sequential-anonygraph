from tqdm import tqdm
from glob import glob
import argparse
import logging
import os
import itertools
from joblib import Parallel, delayed

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.time_graph_generators as generators

logger = logging.getLogger(__file__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_sequence_data_argument(parser)
    rutils.add_info_loss_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)

def generate_pair_distances_for_graph(data_name, sample, strategy, time_instance, info_loss_name, sensitive_attr, args):
    # load subgraph
    graph = dutils.load_raw_subgraph(data_name, sample, strategy, time_instance, args)
    logger.debug(graph)


    # load info loss
    info_loss_fn = utils.get_info_loss_function(info_loss_name, graph, args)

    # for each pair of users, calculate the distance and add write it to a file
    entity_ids = graph.entity_ids

    pair_dist_path = putils.get_pair_distance_of_subgraph_path(data_name, sample, strategy, time_instance, info_loss_name, args)
    logger.debug(pair_dist_path)
    if not os.path.exists(os.path.dirname(pair_dist_path)):
        logger.info("creating folder: {}".format(os.path.dirname(pair_dist_path)))
        os.makedirs(os.path.dirname(pair_dist_path))
    else:
        file_paths = glob(os.path.join(pair_dist_path, "*.*"))
        logger.info("removing {} existing paths in {}".format(len(file_paths), pair_dist_path))

        for path in file_paths:
            os.remove(path)

    logger.info("writing to {}".format(pair_dist_path))
    with open(pair_dist_path, "w+") as f:
        for entity1_id, entity2_id in itertools.product(entity_ids, entity_ids):
            if entity1_id < entity2_id:
                info_loss_value = info_loss_fn.call({entity1_id, entity2_id})

                line = "{},{},{}".format(entity1_id, entity2_id, info_loss_value)
                f.write("{}\n".format(line))


def main(args):
    logger.info(args)
    # list all graphs
    raw_sequence_path = putils.get_sequence_raw_subgraphs_path(args["data"], args["sample"], args["strategy"], args)
    folders = os.listdir(raw_sequence_path)
    logger.debug(folders)
    time_instances = []
    for folder in folders:
        path = os.path.join(raw_sequence_path, folder)
        if os.path.isdir(path):
            time_instances.append(int(folder))

    # folders = list(filter(os.path.isdir, paths))
    # logger.debug(folders)
    # time_instances = [int(folder) for folder in folders]
    logger.info("generating pairwise distances of time instances: {}".format(time_instances))
    # time_instances = [49]
    # for each graph, generate dist and write it
    Parallel(n_jobs=args["workers"])(delayed(generate_pair_distances_for_graph)(args["data"], args["sample"], args["strategy"], t, args["info_loss"], dutils.get_sensitive_attribute_name(args["data"], args["sattr"]), args) for t in tqdm(time_instances))
    # for t in time_instances:
    #     logger.info("generating time instance for: {}".format(t))
    #     generate_pair_distances_for_graph(args["data"], args["sample"], args["strategy"], t, args["info_loss"], args)


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
