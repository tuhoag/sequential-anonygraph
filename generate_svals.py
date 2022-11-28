import time
import argparse
import logging
from joblib import Parallel, delayed
import numpy as np

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


def read_time_groups_from_file(path):
    time_groups = {}

    with open(path, "r") as f:
        for line_idx, line in enumerate(f):
            splits = line.strip().split(",")
            time_groups[line_idx] = [int(t_str) for t_str in splits]

    return time_groups


def generate_svals(graph, sattr_name, time_groups):
    sattr_id = graph.get_relation_id(sattr_name, ATTRIBUTE_RELATION_TYPE)

    all_sval_ids = list(graph.get_domain_value_ids(sattr_id))

    non_sattr_entity_ids = {}
    entity2sval = {}

    for t, insts in time_groups.items():
        # find users who dont have s attr
        all_entity_ids = set()
        having_sattr_entity_ids = set()


        for inst in insts:
            all_entity_ids.update(graph.get_entities_iter(inst))

            for entity_id, _, value_id in graph.get_relation_edges_iter(inst, sattr_id):
                current_sval_ids = entity2sval.get(entity_id, set())
                current_sval_ids.add(value_id)
                entity2sval[entity_id] = current_sval_ids

                having_sattr_entity_ids.add(entity_id)


        current_non_sattr_entity_ids = all_entity_ids.difference(having_sattr_entity_ids)
        non_sattr_entity_ids[t] = current_non_sattr_entity_ids

    t2extra_sval_edges = {}

    entity2sval_list = {}
    for entity_id in entity2sval.keys():
        entity2sval_list[entity_id] = list(entity2sval[entity_id])

    for t, current_non_sattr_entity_ids in non_sattr_entity_ids.items():
        current_extra_edges = []
        for entity_id in current_non_sattr_entity_ids:
            sval_ids = entity2sval_list.get(entity_id)

            if sval_ids is None:
                sval_id = np.random.choice(all_sval_ids)

                entity2sval[entity_id] = {sval_id}
            else:
                sval_id = np.random.choice(sval_ids)

            # add edges
            current_extra_edges.append((entity_id, sval_id))

        t2extra_sval_edges[t] = current_extra_edges

    return t2extra_sval_edges

def write_extra_svals(extra_svals, path):
    with open(path, "w") as f:
        for t, extra_edges in extra_svals.items():
            f.write("{} {}\n".format(t, len(extra_edges)))

            for entity_id, sval_id in extra_edges:
                f.write("{},{}\n".format(entity_id, sval_id))


def main(args):
    logger.debug(args)
    data_name = args["data"]
    sample = args["sample"]
    strategy_name = args["strategy"]

    sensitive_attr = dutils.get_sensitive_attribute_name(args["data"], args["sattr"])
    logger.info("loading dynamic graph...")
    start_time = time.time()
    graph = dutils.load_dynamic_graph_from_output_file(args['data'], args['sample'])
    logger.info("loaded dynamic graph: {} in {}".format(graph, time.time() - start_time))

    logger.debug("all time instances: {}".format(graph.time_instances))
    logger.info("loading time groups file...")
    start_time = time.time()
    time_groups = dutils.read_time_groups(args["data"], args["sample"], args["strategy"], args)
    logger.info("loaded time group file in {}".format(time.time() - start_time))

    extra_svals = generate_svals(graph, sensitive_attr, time_groups)

    extra_svals_path = putils.get_extra_sval_edges_path(data_name, sample, strategy_name, args)
    write_extra_svals(extra_svals, extra_svals_path)


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
