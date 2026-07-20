import argparse

parser = argparse.ArgumentParser(description='DouZero: PyTorch DouDizhu AI')

# General Settings
parser.add_argument('--xpid', default='douzero',
                    help='Experiment id (default: douzero)')
parser.add_argument('--save_interval', default=30, type=int,
                    help='Time interval (in minutes) at which to save the model')
parser.add_argument('--checkpoint_sidecar_retention', default=2, type=int,
                    help='Per-role eval sidecars to retain: 0 disables, -1 keeps all')
parser.add_argument('--objective', default='adp', type=str, choices=['adp', 'wp', 'logadp'],
                    help='Use ADP or WP as reward (default: ADP)')

# Training settings
# Item 5: boolean flags use BooleanOptionalAction so that a YAML ``true`` can be
# overridden to ``false`` from the CLI via ``--no-<flag>``. The positive form
# ``--<flag>`` is unchanged from the legacy ``store_true`` behavior (sets True),
# so existing scripts and docs keep working. Default remains False for all four.
parser.add_argument('--actor_device_cpu', action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Use CPU as actor device (--no-actor_device_cpu forces False)')
parser.add_argument('--gpu_devices', default='0', type=str,
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=1, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=5, type=int,
                    help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='0', type=str,
                    help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Load an existing model (--no-load_model forces False)')
parser.add_argument('--disable_checkpoint', action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Disable saving checkpoint (--no-disable_checkpoint forces False)')
parser.add_argument('--savedir', default='douzero_checkpoints',
                    help='Root dir where experiment data will be saved')

# Hyperparameters
parser.add_argument('--total_frames', default=100000000000, type=int,
                    help='Total frames; must be divisible by unroll_length * batch_size')
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

# P14 system controls. Numerical/performance features are opt-in; the safe
# versioned actor snapshot path is always used.
parser.add_argument('--sync_interval_updates', default=1, type=int,
                    help='Publish actor policy after this many learner updates')
parser.add_argument('--policy_snapshot_slots', default=2, type=int,
                    help='Shared immutable actor-policy slots (minimum 2)')
parser.add_argument('--amp_enabled', action=argparse.BooleanOptionalAction,
                    default=False, help='Enable learner autocast/GradScaler')
parser.add_argument('--amp_dtype', choices=['float16', 'bfloat16'],
                    default='float16', help='AMP autocast dtype')
parser.add_argument('--amp_fallback_on_nonfinite',
                    action=argparse.BooleanOptionalAction, default=True,
                    help='Retry an anomalous AMP step once in float32')
parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction,
                    default=False, help='Pin CPU learner batches before transfer')
parser.add_argument('--ddp_enabled', action=argparse.BooleanOptionalAction,
                    default=False, help='Enable torchrun DistributedDataParallel')
parser.add_argument('--ddp_backend', choices=['auto', 'nccl', 'gloo'],
                    default='auto', help='DDP process-group backend')
parser.add_argument('--compile_model', action=argparse.BooleanOptionalAction,
                    default=False, help='Enable torch.compile after benchmarking')

# Opt-in V1/Legacy performance controls. ``legacy_actor_backend=legacy`` and
# the disabled data-path/compiler flags preserve the original actor/learner
# contract. Factorized actors default to one intra-op thread because every
# actor is its own process; legacy actors keep their historical thread setting.
parser.add_argument('--legacy_actor_backend',
                    choices=['legacy', 'factorized', 'centralized_factorized'],
                    default='legacy', help='V1 actor inference backend')
parser.add_argument('--actor_torch_threads', default=0, type=int,
                    help='Actor Torch threads; 0 preserves legacy (factorized defaults to 1)')
parser.add_argument('--legacy_contiguous_buffers',
                    action=argparse.BooleanOptionalAction, default=False,
                    help='Use contiguous [num_buffers,T,...] rollout storage')
parser.add_argument('--legacy_bulk_rollout',
                    action=argparse.BooleanOptionalAction, default=False,
                    help='Copy complete actor unrolls instead of Python timestep writes')
parser.add_argument('--legacy_flush_ge',
                    action=argparse.BooleanOptionalAction, default=False,
                    help='Submit a rollout as soon as exactly T transitions exist')
parser.add_argument('--legacy_reusable_pinned_staging',
                    action=argparse.BooleanOptionalAction, default=False,
                    help='Reuse pinned CPU batch staging for nonblocking H2D')
parser.add_argument('--legacy_log_interval_seconds', default=0.0, type=float,
                    help='0 logs every learner step; positive values log from the monitor')
parser.add_argument('--legacy_monitor_interval_seconds', default=5.0, type=float,
                    help='Main monitor and throughput reporting cadence')
parser.add_argument('--legacy_profile', action=argparse.BooleanOptionalAction,
                    default=False, help='Collect detailed V1 actor/learner timings')
parser.add_argument('--legacy_profile_sample_interval', default=10, type=int,
                    help='Profile one learner update per this many updates')
parser.add_argument('--legacy_metrics_path', default='', type=str,
                    help='Optional final V1 benchmark/profiler JSON path')
parser.add_argument('--benchmark_warmup_frames', default=0, type=int,
                    help='Frames excluded from the metrics measurement window')
parser.add_argument('--compile_actor', action=argparse.BooleanOptionalAction,
                    default=False, help='Compile actor only (unsupported for V1 dynamic actions)')
parser.add_argument('--compile_learner', action=argparse.BooleanOptionalAction,
                    default=False, help='Compile the fixed-shape V1 learner forward')
parser.add_argument('--rmsprop_foreach', action=argparse.BooleanOptionalAction,
                    default=False, help='Use the RMSprop foreach implementation')
parser.add_argument('--grad_clip_foreach', action=argparse.BooleanOptionalAction,
                    default=False, help='Use foreach gradient clipping')
parser.add_argument('--central_actor_max_actions', default=512, type=int,
                    help='Per-request action capacity for centralized V1 inference')
parser.add_argument('--central_actor_microbatch', default=4, type=int,
                    help='Deprecated old-C0 target microbatch compatibility knob')
parser.add_argument('--central_actor_envs_per_actor', default=4, type=int,
                    help='Independent C0 games interleaved by each actor')
parser.add_argument('--central_actor_min_microbatch', default=2, type=int,
                    help='Minimum C0 microbatch before the delay expires')
parser.add_argument('--central_actor_target_microbatch', default=8, type=int,
                    help='Adaptive C0 target microbatch')
parser.add_argument('--central_actor_max_microbatch', default=16, type=int,
                    help='Maximum C0 microbatch and reusable staging capacity')
parser.add_argument('--central_actor_max_delay_ms', default=2.0, type=float,
                    help='Maximum centralized inference queue delay')
parser.add_argument('--central_actor_max_pending_requests', default=128, type=int,
                    help='Bounded global C0 inference request queue capacity')
parser.add_argument('--central_actor_queue_high_watermark', default=32, type=int,
                    help='Queue depth that throttles new learner updates')
parser.add_argument('--central_actor_inference_deadline_ms', default=10.0, type=float,
                    help='Oldest request age that throttles new learner updates')
parser.add_argument('--central_actor_learner_throttle',
                    action=argparse.BooleanOptionalAction, default=False,
                    help='Delay new learner updates under C0 inference pressure')
parser.add_argument('--central_actor_learner_throttle_mode',
                    choices=['off', 'fixed_threshold', 'predicted_drain_time'],
                    default='fixed_threshold',
                    help='C0 learner admission policy when throttling is enabled')
parser.add_argument('--central_actor_predicted_drain_target_ms', default=10.0,
                    type=float, help='Predicted C0 backlog drain-time target')
parser.add_argument('--central_actor_use_stream_priority',
                    action=argparse.BooleanOptionalAction, default=True,
                    help='Request a high-priority CUDA inference stream')
parser.add_argument('--central_actor_async_policy_copy',
                    action=argparse.BooleanOptionalAction, default=True,
                    help='Copy C0 policy snapshots on a separate CUDA stream')
parser.add_argument('--central_actor_runtime', choices=['process', 'thread'],
                    default='thread',
                    help='Run centralized inference in a CUDA process or main-process thread')
parser.add_argument('--central_actor_split_dense1',
                    action=argparse.BooleanOptionalAction, default=False,
                    help='Split shared/action dense1 projection for C0 inference')
parser.add_argument('--central_actor_staging_dtype', choices=['float32', 'int8'],
                    default='float32', help='Pinned C0 request staging dtype')
parser.add_argument('--central_actor_timeout_seconds', default=30.0, type=float,
                    help='Actor timeout while waiting for centralized inference')

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
parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Force deterministic torch algorithms (P01). Off by default '
                         '(--no-deterministic forces False).')
parser.add_argument('--feature_version', default='legacy', type=str, choices=['legacy', 'v2'],
                    help='Observation feature version. "legacy" (default) is the original '
                         'encoder. "v2" (P03) enables the versioned PublicObservation/observation '
                         'V2 schema; it is accepted but not yet wired into training until P05/P06.')
parser.add_argument('--ruleset', default='legacy', type=str, choices=['legacy', 'standard'],
                    help='Rule set identifier (P02). "legacy" (default) reproduces the '
                         'original environment. "standard" adds bidding/scoring but is '
                         'not yet supported for training.')
parser.add_argument('--model_version', default='legacy', type=str, choices=['legacy', 'factorized', 'v2'],
                    help='Model version. "legacy" (default) is the original per-action forward. '
                         '"factorized" (P04) encodes the shared state/history once per decision and '
                         'is numerically equivalent to legacy under the same weights; it is a '
                         'DEPLOYMENT-only optimization (DeepAgent backend=legacy_factorized). '
                         '"v2" (P05) is the shared state/action model with role embeddings and '
                         'multi-head outputs (DeepAgentV2 backend=v2); it requires '
                         'feature_version=v2. Training is NOT yet wired to factorized/v2; the '
                         'training gate in dmc.py rejects them until P06.')


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
        # BooleanOptionalAction: re-register with default=SUPPRESS so both
        # --flag (True) and --no-flag (False) appear only when the user typed
        # them. This is what lets a CLI --no-<flag> override a YAML true.
        #
        # IMPORTANT: a BooleanOptionalAction's option_strings already contains
        # BOTH "--flag" and "--no-flag". Passing both to add_argument() would
        # make BooleanOptionalAction generate "--no-no-flag" as well (it derives
        # the negation from each registered option). We must re-register ONLY
        # the positive form(s); BooleanOptionalAction re-derives the negation.
        if isinstance(action, _ap.BooleanOptionalAction):
            positive_options = [
                opt for opt in action.option_strings
                if not opt.startswith("--no-")
            ]
            op.add_argument(*positive_options,
                            action=_ap.BooleanOptionalAction,
                            dest=action.dest, default=_ap.SUPPRESS)
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
            # store_true / store_false (legacy non-BooleanOptionalAction flags)
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
