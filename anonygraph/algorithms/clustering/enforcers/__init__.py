
from anonygraph.constants import *
from .base_enforcer import BaseEnforcer
from .invalid_removal_enforcer import InvalidRemovalEnforcer
from .merge_split_assignment_enforcer import MergeSplitAssignmentEnforcer
from .split_overlap_assignmnet_enforcer import SplitOverlapAssignmentEnforcer
from .greedy_split_enforcer import GreedySplitEnforcer
from .icde_enforcer import ICDEEnforcer

def get_enforcer(enforcer_name, args):
    # baseline
    if INVALID_REMOVAL_ENFORCER == enforcer_name:
        min_size = args["k"]
        min_signature_size = args["l"]

        enforcer_fn = InvalidRemovalEnforcer(min_size, min_signature_size)

    elif MERGE_SPLIT_ASSIGNMENT_ENFORCER == enforcer_name:
        min_size = args["k"]
        min_signature_size = args["l"]
        max_dist = args["max_dist"]

        enforcer_fn = MergeSplitAssignmentEnforcer(min_size, min_signature_size, max_dist)

    elif SPLIT_OVERLAP_ASSIGNMENT_ENFORCER == enforcer_name:
        min_size = args["k"]
        min_signature_size = args["l"]
        max_dist = args["max_dist"]

        enforcer_fn = SplitOverlapAssignmentEnforcer(min_size, min_signature_size, max_dist)
    # is using
    elif GREEDY_SPLIT_ENFORCER == enforcer_name:
        min_size = args["k"]
        min_signature_size = args["l"]
        max_dist = args["max_dist"]

        enforcer_fn = GreedySplitEnforcer(min_size, min_signature_size, max_dist)

    elif ICDE_ENFORCER == enforcer_name:
        min_size = args["k"]
        max_dist = args["max_dist"]

        enforcer_fn = ICDEEnforcer(min_size, 1, max_dist)
    else:
        raise Exception("Unsupported enforcer '{}'. Only supported {}".format(enforcer_name, [INVALID_REMOVAL_ENFORCER, MERGE_SPLIT_ASSIGNMENT_ENFORCER]))

    return enforcer_fn