import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Distribution
import torch_ac
import math

from torch.distributions.utils import _standard_normal, lazy_property

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.feature_size = 128
        self.feature_extractor = nn.Sequential(
            nn.Linear(4, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size),
            nn.ReLU(),
        )

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, 3),
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, 1),
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.feature_size

    def forward(self, obs, memory):
        x = self.feature_extractor(obs.state)
        embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


class AdversaryACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(
        self,
        env,
        action_std_init=0.4,
        device="cuda",
        action_std_decay_rate=0.025,
        min_action_std=0.2,
    ):
        super().__init__()

        self.action_dim = env.state_size
        self.device = device

        # Define feature extractor
        self.feature_size = 128
        self.feature_extractor = nn.Sequential(
            nn.Linear(env.state_size + 1, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size),
            nn.ReLU(),
        )

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.Tanh(),
            nn.Linear(self.feature_size, self.action_dim),
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.Tanh(),
            nn.Linear(self.feature_size, 1),
        )

        # Initialize parameters correctly
        self.apply(init_params)

        self.action_var = torch.full((self.action_dim,), action_std_init ** 2).to(
            self.device
        )

        self.action_std = action_std_init
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std

    def _get_action_dist(self, embedding, old_probas, budget):
        action_mean = self.actor(embedding).exp()
        action_budget = torch.ones_like(action_mean) * budget
        action_mean = torch.minimum(action_mean, action_budget)

        mask = old_probas > 0

        if len(old_probas) == 1:
            action_var = torch.diag(self.action_var)
        else:
            action_var = torch.diag_embed(self.action_var).unsqueeze(0)

        return MaskedMultivariateNormal(action_mean, mask, scale_tril=action_var)

    def forward(self, obs, memory):
        budget = obs.remaining_budget

        x = self.feature_extractor(obs.transition_probas)
        embedding = x

        dist = self._get_action_dist(embedding, obs.transition_probas[:, :-1], budget)
        perturbations = dist.sample()
        perturbations = torch.where(
            perturbations < 0, dist.loc, perturbations
        )  # Readjust when neg probas

        budget_array = torch.ones_like(dist.loc) * budget
        perturbations = torch.where(perturbations > budget, budget_array, perturbations)

        x = self.critic(embedding)
        value = x.squeeze(1)

        return perturbations, dist, value, memory

    def update_action_std(self):
        self.action_std = self.action_std - self.action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= self.min_action_std:
            self.action_std = self.min_action_std

        self.action_var = self.action_var = torch.full(
            (self.action_dim,), self.action_std ** 2
        ).to(self.device)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.feature_size


def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)


def _masked_batch_mahalanobis(bL, bx, mask):
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for (sL, sx) in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (
        list(range(outer_batch_dims))
        + list(range(outer_batch_dims, new_batch_dims, 2))
        + list(range(outer_batch_dims + 1, new_batch_dims, 2))
        + [new_batch_dims]
    )
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = (
        torch.triangular_solve(flat_x_swap, flat_L, upper=False)[0].pow(2) * mask.t()
    ).sum(
        -2
    )  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)


def _precision_to_scale_tril(P):
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    Lf = torch.cholesky(torch.flip(P, (-2, -1)))
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    L = torch.triangular_solve(
        torch.eye(P.shape[-1], dtype=P.dtype, device=P.device), L_inv, upper=False
    )[0]
    return L


class MaskedMultivariateNormal(Distribution):
    def __init__(
        self,
        loc,
        mask,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        validate_args=False,
    ):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (
            precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(scale_tril.shape[:-2], loc.shape[:-1])
            self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(
                covariance_matrix.shape[:-2], loc.shape[:-1]
            )
            self.covariance_matrix = covariance_matrix.expand(batch_shape + (-1, -1))
        else:
            if precision_matrix.dim() < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(
                precision_matrix.shape[:-2], loc.shape[:-1]
            )
            self.precision_matrix = precision_matrix.expand(batch_shape + (-1, -1))
        self.loc = loc.expand(batch_shape + (-1,))

        event_shape = self.loc.shape[-1:]
        super(MaskedMultivariateNormal, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = torch.cholesky(covariance_matrix)
        else:  # precision_matrix is not None
            self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)

        self.mask = mask

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self):
        return torch.matmul(
            self._unbroadcasted_scale_tril,
            self._unbroadcasted_scale_tril.transpose(-1, -2),
        ).expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self):
        identity = torch.eye(
            self.loc.size(-1), device=self.loc.device, dtype=self.loc.dtype
        )
        # TODO: use cholesky_inverse when its batching is supported
        return torch.cholesky_solve(identity, self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return (
            self._unbroadcasted_scale_tril.pow(2)
            .sum(-1)
            .expand(self._batch_shape + self._event_shape)
        )

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return (self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)) * self.mask

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _masked_batch_mahalanobis(self._unbroadcasted_scale_tril, diff, self.mask)
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log() * self.mask
        ).sum(-1)
        return -0.5 * (self.mask.sum(-1) * math.log(2 * math.pi) + M) - half_log_det

    def entropy(self):
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log() * self.mask
        ).sum(-1)
        H = 0.5 * self.mask.sum(-1) * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
