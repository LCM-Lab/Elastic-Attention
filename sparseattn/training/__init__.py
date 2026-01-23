from .lh_trainer import Trainer
from .modeling_flash_qwen import PawQwen3ForCausalLM
from .script_arguments import ScriptArguments
Qwen3ForCausalLM = PawQwen3ForCausalLM

__all__ = [
    "Trainer",
    "PawQwen3ForCausalLM",
    "ScriptArguments",
    # Aliases for backward compatibility
    "Qwen3ForCausalLM",
]
