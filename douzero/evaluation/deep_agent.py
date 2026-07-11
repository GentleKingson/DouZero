import torch
import numpy as np

from douzero.env.env import get_obs

# Supported deployment backends. ``legacy`` is the original per-row forward
# (the LSTM runs once per legal action). ``legacy_factorized`` (P04) encodes
# the shared history/state once per decision and is numerically equivalent to
# ``legacy`` under the same weights; it loads the SAME per-position .ckpt with
# no conversion. The default stays ``legacy`` so existing behavior is unchanged
# until a caller explicitly opts into the factorized path.
SUPPORTED_BACKENDS = ('legacy', 'legacy_factorized')


def _load_model(position, model_path, backend='legacy'):
    """Load a per-position role model for a given backend.

    Both backends consume the same legacy per-position sidecar (a bare
    ``state_dict``): the factorized models declare identical submodule names
    and shapes, so ``load_legacy_position_ckpt`` + the key filter loads either
    without conversion. P16 replaces this permissive filter with a strict
    manifest load.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}; expected one of {SUPPORTED_BACKENDS}."
        )
    from douzero.checkpoint import load_legacy_position_ckpt
    if backend == 'legacy_factorized':
        from douzero.dmc.models_factorized import factorized_model_dict
        model = factorized_model_dict[position]()
    else:
        from douzero.dmc.models import model_dict
        model = model_dict[position]()
    model_state_dict = model.state_dict()
    # Legacy per-position sidecar: bare state_dict. The permissive key filter
    # below is pinned by P00 tests; P16 replaces it with a strict manifest load.
    pretrained = load_legacy_position_ckpt(model_path)
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


class DeepAgent:

    def __init__(self, position, model_path, backend='legacy'):
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}; expected one of {SUPPORTED_BACKENDS}."
            )
        self.backend = backend
        self.model = _load_model(position, model_path, backend=backend)

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs(infoset)

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        return best_action
