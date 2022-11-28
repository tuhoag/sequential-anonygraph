import logging
import copy

from .same_attributes_gen import SameAttributesGenerator
from .unused_entities_removal_gen import UnusedEntitiesRemovalGenerator
from .same_degree_relationships_gen import SameDegreeRelationshipsGenerator
from .same_degree_relationships_gen2 import SameDegreeRelationshipsGenerator2
from .svals_gen import SensitiveValuesGenerator

logger = logging.getLogger(__name__)

class EdgesModificationGeneralization(object):
    def __init__(self, subgraph, degree_gen_algo, fake_entity_manager):
        self.subgraph = subgraph
        self.degree_gen_algo = degree_gen_algo
        self.fake_entity_manager = fake_entity_manager

    def __call__(self, clusters):
        logger.debug("clusters: {}".format(clusters))
        new_subgraph = copy.deepcopy(self.subgraph)

        modification_algos = [
            UnusedEntitiesRemovalGenerator(),
        ]

        if len(clusters) > 0:
            modification_algos.extend([
                SameAttributesGenerator(),
                SensitiveValuesGenerator(self.fake_entity_manager),
            ])

            if self.degree_gen_algo == "old":
                modification_algos.append(SameDegreeRelationshipsGenerator())
            elif self.degree_gen_algo == "new":
                modification_algos.append(SameDegreeRelationshipsGenerator2())
            else:
                raise Exception("Unsupported same degree gen algo: {}".format(self.degree_gen_algo))

        for algo_index, algo_fn in enumerate(modification_algos):
            algo_fn(clusters, new_subgraph)
            logger.debug("step: {} - subgraph: {}".format(algo_index, new_subgraph))

        return new_subgraph