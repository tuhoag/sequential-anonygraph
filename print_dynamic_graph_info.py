import argparse
import logging

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils

logger = logging.getLogger(__file__)

def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_log_argument(parser)



def main(args):
    logger.info(args)
    output_path = putils.get_raw_graph_path(args["data"], args["sample"])
    loaded_graph = dutils.load_dynamic_graph_from_output_file(args["data"], args["sample"])
    logger.info("loaded graph from: {}".format(output_path))
    logger.info("loaded graph: {}".format(loaded_graph))

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)