import argparse
import logging
import os
import itertools

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

    rutils.add_log_argument(parser)
    rutils.add_workers_argument(parser)

def main(args):
    logger.info(args)

    rutils.run_python_file("generate_raw_subgraphs.py", args)
    rutils.run_python_file("generate_pairwise_distances.py", args)


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
