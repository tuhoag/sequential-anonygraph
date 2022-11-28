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

    time_groups = dutils.read_time_groups(args["data"], args["sample"], args["strategy"], args)
    logger.info("start generating {} subgraphs...".format(len(time_groups)))
    start_time = time.time()
    sorted_time_keys = sorted(time_groups.keys())
    Parallel(n_jobs=args["workers"])(
        delayed(rutils.run_generate_raw_subgraph_file)(t, args)
        for t in tqdm(sorted_time_keys)
    )
    logger.info("generated all subgraphs in {}".format(time.time() - start_time))


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
