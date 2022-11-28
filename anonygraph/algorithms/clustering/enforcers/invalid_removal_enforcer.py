from typing import Dict, List, Set
from anonygraph.algorithms.cluster import Cluster
from anonygraph.algorithms.fake_entity_manager import FakeEntityManager
from anonygraph.algorithms.pair_distance import PairDistance
from anonygraph.info_loss import info
from .base_enforcer import BaseEnforcer

class InvalidRemovalEnforcer(BaseEnforcer):
    def __init__(self, min_size, min_signature_size):
        self.min_size = min_size
        self.min_signature_size = min_signature_size


    # clusters, pair_distance, entity2svals, all_sval_ids, fake_entity_manager
    def __call__(self, clusters: List[Cluster], pair_distance: PairDistance, entity2svals: Dict[int, Set[int]], all_sval_ids: Set[int], fake_entity_manager: FakeEntityManager) -> List[Cluster]:
        # pass
    # def __call__(self, clusters, pair_distance, entity2svals, fake_entity_manager):
        new_clusters = []

        for cluster in clusters:
            if len(cluster) < self.min_size or len(info.get_generalized_signature_info_from_dict(entity2svals, fake_entity_manager, cluster)) < self.min_signature_size:
                continue

            new_clusters.append(cluster)

        return new_clusters

    def update(self, clusters, pair_distance, entity2svals, fake_entity_manager, historical_table):
        new_clusters = []

        for cluster in clusters:
            signature = info.get_generalized_signature_info_from_dict(entity2svals, cluster)

            if len(cluster) < self.min_size or len(signature) < self.min_signature_size:
                continue

            new_clusters.append(cluster)

        return new_clusters
