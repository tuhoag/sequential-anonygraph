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


    logger.info("grouping time instances...")
    start_time = time.time()
    rutils.run_generate_time_groups_file(args)
    logger.info("groupped time instances in {}".format(time.time() - start_time))



    logger.info("generating missing svals...")
    start_time = time.time()
    rutils.run_generate_svals_file(args)
    logger.info("generated missing svals in {}".format(time.time() - start_time))


    logger.info("generating subgraphs...")
    start_time = time.time()
    rutils.run_generate_all_raw_subgraphs_file(args)
    logger.info("generated all subgraphs in {}".format(time.time() - start_time))



if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
