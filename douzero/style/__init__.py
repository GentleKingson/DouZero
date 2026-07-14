"""Public other-player style features and neural encoder (P11)."""

from .encoder import StyleEncoder
from .features import (
    STYLE_FEATURE_VERSION,
    STYLE_FEATURE_WIDTH,
    STYLE_LAYOUT_HASH,
    STYLE_NUM_OTHER_PLAYERS,
    STYLE_PER_PLAYER_WIDTH,
    build_style_features,
)

__all__ = [
    "STYLE_FEATURE_VERSION",
    "STYLE_FEATURE_WIDTH",
    "STYLE_LAYOUT_HASH",
    "STYLE_NUM_OTHER_PLAYERS",
    "STYLE_PER_PLAYER_WIDTH",
    "StyleEncoder",
    "build_style_features",
]
