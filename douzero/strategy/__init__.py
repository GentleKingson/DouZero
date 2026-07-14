"""Public-information strategy features and auxiliary supervision (P09).

The package deliberately contains no policy overrides.  Every helper ranks or
describes actions that the rule engine has already declared legal.
"""

from .config import STRATEGY_FEATURE_VERSION, StrategyFeatureConfig
from .features import STRATEGY_FEATURE_NAMES, build_strategy_feature_matrix
from .hand_decomposition import DecompositionResult, hand_decomposition
from .structure import ActionStructureCost, action_structure_cost

__all__ = [
    "ActionStructureCost",
    "DecompositionResult",
    "STRATEGY_FEATURE_NAMES",
    "STRATEGY_FEATURE_VERSION",
    "StrategyFeatureConfig",
    "action_structure_cost",
    "build_strategy_feature_matrix",
    "hand_decomposition",
]
