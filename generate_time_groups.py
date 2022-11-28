import os
import argparse
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
import time

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

def write_time_group_to_file(path, time_groups):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, "w") as f:
        for group_id, time_instances in time_groups.items():
            line = ",".join(map(str, time_instances))
            f.write("{}\n".format(line))


def main(args):
    logger.info(args)
    logger.info("loading dynamic graph...")
    start_time = time.time()
    graph = dutils.load_dynamic_graph_from_output_file(args["data"], args["sample"])
    logger.debug("loaded dynamic graph: {} in {}".format(graph, time.time() - start_time))

    sattr = dutils.get_sensitive_attribute_name(args["data"], args["sattr"])
    if not graph.is_attribute_relation_name(sattr):
        raise Exception("attr '{}' is not existed. Must be among {}".format(sattr, graph.attribute_relation_ids))

    logger.info("grouping time instances...")
    start_time = time.time()
    gen = generators.get_strategy(args["strategy"], args)
    time_groups = gen.run(graph)
    logger.info("groupped time instances in {}".format(time.time() - start_time))

    logger.info("start writing time groups to file")
    start_time = time.time()
    time_groups_path = putils.get_time_group_path(args["data"], args["sample"], args["strategy"], args)
    write_time_group_to_file(time_groups_path, time_groups)
    logger.info("saved writing time groups at: {} in {}".format(time_groups_path, time.time() - start_time))


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
