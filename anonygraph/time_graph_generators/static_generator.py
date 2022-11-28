import logging
import os

from joblib import Parallel, delayed

# from .utils import , test_group_time_instances, generate_graph

logger = logging.getLogger(__name__)

def merge_all_time_insts_to_one_group(graph):
    time_instances = sorted(graph.time_instances)
    logger.debug("sorted time instances: {}".format(time_instances))
    groups = {0: time_instances}
    return groups

class StaticGenerator(object):
    def __init__(self, num_workers=1):
        self.num_workers = num_workers

    def run(self, graph):
        logger.info("start grouping time instances")
        time_groups = merge_all_time_insts_to_one_group(graph)
        logger.info("finished grouping time instances: {}".format(time_groups))

        logger.info(
            'generating {} graphs with {} workers'.format(
                len(time_groups), self.num_workers
            )
        )

        return time_groups