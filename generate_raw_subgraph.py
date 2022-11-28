import time
import argparse
import logging
from joblib import Parallel, delayed

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.time_graph_generators as generators
import anonygraph.constants as constants

logger = logging.getLogger(__file__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_sequence_data_argument(parser)
    rutils.add_log_argument(parser)

    parser.add_argument("--dest_t", type=int)
    parser.add_argument("--src_t_list", type=rutils.string2list(int))


def write_subgraph(t, subgraph, args):
    sequence_path = putils.get_raw_subgraph_path(
        args['data'], args['sample'], args['strategy'], t, args
    )
    logger.info('saving edges to: {}'.format(sequence_path))
    subgraph.to_edge_files(sequence_path)


def read_time_groups_from_file(path):
    time_groups = {}

    with open(path, "r") as f:
        for line_idx, line in enumerate(f):
            splits = line.strip().split(",")
            time_groups[line_idx] = [int(t_str) for t_str in splits]

    return time_groups


def load_extra_svals(data_name, sample, strategy_name, dest_t, args):
    # load
    path = putils.get_extra_sval_edges_path(data_name, sample, strategy_name, args)

    entity2extra_svals = {}
    # count = 0
    with open(path, "r") as f:
        line = f.readline()
        while(line != ""):
            # count += 1
            # line = f.readline().strip()

            splits = line.strip().split(" ")
            # logger.debug("count: {} - line={}, splits (len={}): {}".format(count, line, len(splits), splits))

            if len(splits) == 2:
                # t, num_edges = [int(item) for item in f.readline().strip().split(" ")]
                t, num_edges = map(int, splits)
                # logger.debug("t={}, num_edges={}".format(t, num_edges))
                # raise Exception()
                if t == dest_t:
                    for _ in range(num_edges):
                        line = f.readline().strip()
                        splits = line.split(",")
                        # logger.debug("inside count: {} - line={}, splits: {}".format(count, line, splits))
                        entity_id, sval_id = map(int, splits)

                        sval_ids = entity2extra_svals.get(entity_id)

                        if sval_ids is None:
                            sval_ids = []
                            entity2extra_svals[entity_id] = sval_ids

                        sval_ids.append(sval_id)
                        # logger.debug("add entity_id:{} - sval_id: {}".format(entity_id, sval_id))

                    # break

            # if count >= 20:
            #     raise Exception()

            line = f.readline()

    return entity2extra_svals

def main(args):
    logger.debug(args)
    data_name = args["data"]
    sample = args["sample"]
    strategy_name = args["strategy"]

    sensitive_attr = dutils.get_sensitive_attribute_name(
        args["data"], args["sattr"]
    )
    logger.info("loading dynamic graph...")
    start_time = time.time()
    graph = dutils.load_dynamic_graph_from_output_file(
        args['data'], args['sample']
    )
    logger.debug(
        "loaded dynamic graph: {} in {}".format(
            graph,
            time.time() - start_time
        )
    )

    logger.debug("all time instances: {}".format(graph.time_instances))
    logger.info("loading time groups file...")
    start_time = time.time()
    time_groups = read_time_groups_from_file(
        putils.get_time_group_path(
            args["data"], args["sample"], args["strategy"], args
        )
    )
    logger.info("loaded time group file in {}".format(time.time() - start_time))

    dest_t = args["dest_t"]
    src_t_list = time_groups[dest_t]
    logger.debug(
        "generating snapshot at t={} for time instances".format(dest_t)
    )

    logger.info("reading all extra svals...")
    entity2extra_svals = load_extra_svals(data_name, sample, strategy_name, dest_t, args)

    logger.info("generating subgraph at time: {}...".format(dest_t))
    start_time = time.time()
    subgraph = graph.generate_subgraph_from_time_instances(
        src_t_list, sensitive_attr, entity2extra_svals
    )
    logger.info(
        "generated subgraph at time {} in {}".format(
            dest_t,
            time.time() - start_time
        )
    )

    logger.info("writing subgraph at time {}...".format(dest_t))
    start_time = time.time()
    write_subgraph(dest_t, subgraph, args)
    logger.info(
        "wrote subgraph at time {} in {}".format(
            dest_t,
            time.time() - start_time
        )
    )


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
