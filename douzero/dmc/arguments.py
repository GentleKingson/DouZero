import argparse

parser = argparse.ArgumentParser(description='DouZero: PyTorch DouDizhu AI')

# General Settings
parser.add_argument('--xpid', default='douzero',
                    help='Experiment id (default: douzero)')
parser.add_argument('--save_interval', default=30, type=int,
                    help='Time interval (in minutes) at which to save the model')    
parser.add_argument('--objective', default='adp', type=str, choices=['adp', 'wp', 'logadp'],
                    help='Use ADP or WP as reward (default: ADP)')    

# Training settings
parser.add_argument('--actor_device_cpu', action='store_true',
                    help='Use CPU as actor device')
parser.add_argument('--gpu_devices', default='0', type=str,
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=1, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=5, type=int,
                    help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='0', type=str,
                    help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='douzero_checkpoints',
                    help='Root dir where experiment data will be saved')

# Hyperparameters
parser.add_argument('--total_frames', default=100000000000, type=int,
                    help='Total environment frames to train for')
parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='The probability for exploration')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Learner batch size')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=50, type=int,
                    help='Number of shared-memory buffers')
parser.add_argument('--num_threads', default=4, type=int,
                    help='Number learner threads')
parser.add_argument('--max_grad_norm', default=40., type=float,
                    help='Max norm of gradients')

# Optimizer settings
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')
parser.add_argument('--epsilon', default=1e-5, type=float,
                    help='RMSProp epsilon')

# P01 additions (optional; defaults preserve legacy behavior). These are
# appended so the original 23 flags and their defaults are unchanged.
parser.add_argument('--config', default='', type=str,
                    help='Optional path to a YAML config file (P01). CLI flags override the file.')
parser.add_argument('--seed', default=0, type=int,
                    help='Base RNG seed (P01). 0 = legacy behavior (unseeded).')
parser.add_argument('--deterministic', action='store_true',
                    help='Force deterministic torch algorithms (P01). Off by default.')
parser.add_argument('--feature_version', default='legacy', type=str, choices=['legacy'],
                    help='Observation feature version (P01). Only "legacy" is supported in P01.')
parser.add_argument('--ruleset', default='legacy', type=str, choices=['legacy'],
                    help='Rule set identifier (P01). Only "legacy" is supported in P01.')
parser.add_argument('--model_version', default='legacy', type=str, choices=['legacy'],
                    help='Model version (P01). Only "legacy" is supported in P01.')


def _build_override_parser():
    """Build a parser whose defaults are SUPPRESS, for detecting explicit CLI flags.

    When `--config` is used, we re-parse with this parser so that ONLY flags the
    user actually typed appear in the Namespace. This cleanly solves the
    "argparse default vs explicit value" ambiguity (including store_true): an
    absent flag simply does not appear, so it never clobbers a YAML value.
    """
    import argparse as _ap

    op = _ap.ArgumentParser(add_help=False)
    for action in parser._actions:
        if action.dest == "help":
            continue
        # Re-register each option string with default=SUPPRESS.
        kwargs = {"default": _ap.SUPPRESS}
        if action.type is not None:
            kwargs["type"] = action.type
        if action.choices:
            kwargs["choices"] = action.choices
        if action.const is not None and not action.option_strings:
            continue  # positional; skip (we have none)
        if action.nargs == 0:
            # store_true / store_false
            op.add_argument(*action.option_strings, action="store_true",
                            dest=action.dest, default=_ap.SUPPRESS)
        else:
            op.add_argument(*action.option_strings, dest=action.dest, **kwargs)
    return op


def parse_args(argv=None):
    """Parse args with optional YAML config support (P01).

    Behavior:
      - No ``--config``: identical to ``parser.parse_args()`` (legacy path).
      - With ``--config <path>``: load the YAML as the base, then overlay ONLY
        the CLI flags the user explicitly typed on top (CLI wins). This is done
        by re-parsing with default=SUPPRESS, so absent flags (including
        store_true defaults) never clobber YAML values.

    The returned object is always an argparse Namespace, so ``train(flags)``
    is unchanged. PyYAML is imported lazily so plain ``--help`` never requires it.
    """
    import sys

    if argv is None:
        argv = sys.argv[1:]
    flags = parser.parse_args(argv)
    if not flags.config:
        return flags
    # Re-parse with SUPPRESS defaults to detect which flags were explicit.
    override_parser = _build_override_parser()
    explicit_ns, _unknown = override_parser.parse_known_args(argv)
    # Load YAML base and overlay only the explicit CLI overrides.
    from douzero.config import load_config, merge, to_argparse_namespace

    base = load_config(flags.config)
    merged = merge(base, explicit_ns)
    return to_argparse_namespace(merged)
