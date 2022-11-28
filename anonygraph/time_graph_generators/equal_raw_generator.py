import logging
import copy
from joblib import Parallel, delayed

from tqdm import tqdm

from .utils import test_group_time_instances, group_equal_size_time_instances

logger = logging.getLogger(__name__)


class EqualRawSizeGenerator(object):
    def __init__(self, num_subgraphs, num_workers=1):
        self.num_subgraphs = num_subgraphs
        self.num_workers = num_workers

    def run(self, graph):
        logger.info("start grouping time instances")
        time_groups = group_equal_size_time_instances(graph, self.num_subgraphs)
        logger.info("finished grouping time instances: {}".format(time_groups))

        test_group_time_instances(time_groups, graph)

        logger.info(
            'generating {} graphs with {} workers'.format(
                len(time_groups), self.num_workers
            )
        )

        return time_groups