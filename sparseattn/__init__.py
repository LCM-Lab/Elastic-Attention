# Core sparse attention implementations
from .src import (
    Xattention_prefill_dim3,
    Xattention_prefill_dim4,
)

# For backward compatibility and ease of use
Xattention = Xattention_prefill_dim4

__all__ = [
    # Aliases for backward compatibility
    "Xattention",
]

__version__ = "0.1.0"
__author__ = "****"
