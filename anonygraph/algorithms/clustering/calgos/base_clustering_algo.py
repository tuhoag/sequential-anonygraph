from abc import ABC, abstractmethod

from anonygraph.algorithms import PairDistance

class BaseClusteringAlgo(ABC):
    @abstractmethod
    def run(self, entity_ids, pair_distance: PairDistance):
        pass