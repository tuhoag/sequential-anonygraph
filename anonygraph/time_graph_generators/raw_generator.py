import logging
import os

from joblib import Parallel, delayed

# from .utils import , test_group_time_instances, generate_graph

logger = logging.getLogger(__name__)

def generate_raw_time_groups(graph):
    time_instances = sorted(graph.time_instances)
    logger.debug("sorted time instances: {}".format(time_instances))

    groups = {}

    for t_idx, t in enumerate(time_instances):
        groups[t_idx] = [t]

    return groups

class RawGenerator(object):
    def __init__(self, num_workers=1):
        self.num_workers = num_workers

    def run(self, graph):
        logger.info("start grouping time instances")
        time_groups = generate_raw_time_groups(graph)
        logger.info("finished grouping time instances: {}".format(time_groups))

        logger.info(
            'generating {} graphs with {} workers'.format(
                len(time_groups), self.num_workers
            )
        )

        return time_groups