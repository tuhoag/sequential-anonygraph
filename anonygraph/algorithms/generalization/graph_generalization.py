import logging

import anonygraph.constants as constants
from .edges_modification_generalization import EdgesModificationGeneralization

logger = logging.getLogger(__name__)
class GraphGeneralization(object):
    def __init__(self, algo, fake_entity_manager=None):
        self.algo = algo
        self.fake_entity_manager = fake_entity_manager

    def run(self, subgraph, clusters):
        logger.debug("clusters: {}".format(clusters))

        # if len(clusters) == 0:
        #     raise Exception()
        if self.algo == constants.ADD_REMOVE_EDGES_GEN:
            gen_fn = EdgesModificationGeneralization(subgraph, "old")
        elif self.algo == constants.ADD_REMOVE_EDGES2_GEN:
            gen_fn = EdgesModificationGeneralization(subgraph, "new", self.fake_entity_manager)
        else:
            raise NotImplementedError("Unsupported generalization algo: {}".format(self.algo))

        anony_subgraph = gen_fn(clusters)
        return anony_subgraph