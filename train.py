import os

from douzero.dmc import train
from douzero.dmc.arguments import parse_args
from douzero.runtime import maybe_set_global_deterministic, set_global_seed

if __name__ == '__main__':
    flags = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    # P01: opt-in seeding. seed=0 (default) is a no-op -> legacy behavior.
    set_global_seed(getattr(flags, "seed", 0))
    maybe_set_global_deterministic(getattr(flags, "deterministic", False))
    train(flags)
