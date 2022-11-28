import logging
import copy
from joblib import Parallel, delayed

from tqdm import tqdm

from .utils import test_group_time_instances, group_time_instances_by_mean_edges

logger = logging.getLogger(__name__)


def merge_for_addition(groups):
    time_instances = sorted(list(map(int, groups.keys())))

    logger.debug(time_instances)

    last_group = []
    logger.debug("before: {}".format(groups))
    for t in time_instances:
        group = groups.get(t)
        group.extend(last_group)

        last_group = group

    logger.debug("after: {}".format(groups))

class MeanEdgesStrategy(object):
    def __init__(self, num_subgraphs, num_workers=1):
        self.num_subgraphs = num_subgraphs
        self.num_workers = num_workers

    def run(self, graph):
        # group set of periods
        time_groups = group_time_instances_by_mean_edges(graph, self.num_subgraphs)

        logger.debug(
            "number of groups: {}".format(
                len(time_groups)
            )
        )
        test_group_time_instances(time_groups, graph)

        # merge_for_addition(time_groups)

        # logger.debug(list(time_groups.keys()))

        # for each group, create a graph that contain all edges of that period
        # logger.info(
        #     'generating {} graphs with {} workers'.format(
        #         len(time_groups), self.num_workers
        #     )
        # )

        # raise Exception()

        return time_groups