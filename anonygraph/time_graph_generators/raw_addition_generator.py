import logging
import os

from joblib import Parallel, delayed

# from .utils import , test_group_time_instances, generate_graph

logger = logging.getLogger(__name__)

def generate_raw_addition_time_groups(graph, max_time_instances):
    time_instances = sorted(graph.time_instances)
    logger.debug("sorted time instances: {}".format(time_instances))

    groups = {}

    for t_idx, t in enumerate(time_instances):
        if t_idx >= max_time_instances and max_time_instances != -1:
            break

        previous_time_insts = groups.get(t_idx - 1, [])
        current_time_insts = previous_time_insts.copy()
        current_time_insts.append(t)

        groups[t_idx] = current_time_insts

        logger.debug(groups)

    return groups

class RawAdditionGenerator(object):
    def __init__(self, max_time_instances, num_workers=1):
        self.num_workers = num_workers
        self.max_time_instances = max_time_instances

    def run(self, graph):
        logger.info("start grouping time instances")
        time_groups = generate_raw_addition_time_groups(graph, self.max_time_instances)
        logger.info("finished grouping time instances: {}".format(time_groups))
        # raise Exception()

        return time_groups