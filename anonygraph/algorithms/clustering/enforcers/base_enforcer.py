from abc import ABC, abstractmethod
from anonygraph.algorithms.history_table import HistoryTable
from anonygraph.algorithms.fake_entity_manager import FakeEntityManager
from anonygraph.algorithms.pair_distance import PairDistance
from typing import List, Dict, Set

from anonygraph.algorithms import Cluster


class BaseEnforcer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, clusters: List[Cluster], pair_distance: PairDistance, entity2svals: Dict[int, Set[int]], all_sval_ids: Set[int], fake_entity_manager: FakeEntityManager) -> List[Cluster]:
        pass

    @abstractmethod
    def update(self, clusters: List[Cluster], pair_distance: PairDistance, entity2svals: Dict[int, Set[int]], fake_entity_manager: FakeEntityManager, historical_table: HistoryTable) -> List[Cluster]:
        pass