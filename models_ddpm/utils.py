import torch
import numpy as np
import math
import sympy as sym
import json
from typing import Callable
from torch_geometric.nn.inits import glorot_orthogonal
from torch.nn import Linear
from torch import Tensor
from torch_scatter import scatter, segment_csr
from ase.io import read, write
MAX_ATOMIC_NUM = 100

def coord2diff(x, edge_index, norm_constant=1):
    i, j = edge_index
    coord_diff = x[i] - x[j]
    dist = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(dist + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)

    return dist, coord_diff

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)

def sample_centered_gaussian(size, n_mask=None):
    x = torch.randn(size, device=n_mask.device)
    N = torch.sum(n_mask, dim=1, keepdim=True)
    mean = torch.sum(x, dim=1, keepdim=True) / N

    x -= mean
    if n_mask is not None: x *= n_mask
    return x

def sample_gaussian(size, mu=0, sigma=1, n_mask=None, device=None):
    x = mu + sigma * torch.randn(size, device=device)
    if n_mask is not None: x *= n_mask
    return x

def sample_combined_guassian(samples, max_atoms, n_mask, x_dim, h_dim):
    z_x = sample_centered_gaussian(size=(samples, max_atoms, x_dim), n_mask=n_mask)
    z_h = sample_gaussian(size=(samples, max_atoms, h_dim), n_mask=n_mask)

    return torch.cat([z_x, z_h], dim=-1)


def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)

def remove_mean(x, n_mask=None):
    # print(x.shape, n_mask.shape)
    N = n_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    # print(N.shape, mean.shape, x.shape)
    x = x - mean
    if n_mask is not None: x *= n_mask
    return x


def read_json(path):
    """"""
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    with open(path, "r") as f:
        content = json.load(f)
    return content

def read_value_json(path, key):
    """"""
    content = read_json(path)

    if key in content.keys():
        return content[key]
    else:
        return None


class AutomaticFit:
    """
    All added variables are processed in the order of creation.
    """

    activeVar = None
    queue = None
    fitting_mode = False

    def __init__(self, variable, scale_file, name):
        self.variable = variable  # variable to find value for
        self.scale_file = scale_file
        self._name = name

        self._fitted = False
        self.load_maybe()

        # first instance created
        if AutomaticFit.fitting_mode and not self._fitted:

            # if first layer set to active
            if AutomaticFit.activeVar is None:
                AutomaticFit.activeVar = self
                AutomaticFit.queue = []  # initialize
            # else add to queue
            else:
                self._add2queue()  # adding variables to list fill fail in graph mode

    def reset():
        AutomaticFit.activeVar = None
        AutomaticFit.all_processed = False

    def fitting_completed():
        return AutomaticFit.queue is None

    def set2fitmode():
        AutomaticFit.reset()
        AutomaticFit.fitting_mode = True

    def _add2queue(self):
        # logging.debug(f"Add {self._name} to queue.")
        # check that same variable is not added twice
        for var in AutomaticFit.queue:
            if self._name == var._name:
                raise ValueError(
                    f"Variable with the same name ({self._name}) was already added to queue!"
                )
        AutomaticFit.queue += [self]

    def set_next_active(self):
        """
        Set the next variable in the queue that should be fitted.
        """
        queue = AutomaticFit.queue
        if len(queue) == 0:
            # logging.debug("Processed all variables.")
            AutomaticFit.queue = None
            AutomaticFit.activeVar = None  # reset to None
            return
        AutomaticFit.activeVar = queue.pop(0)

    def load_maybe(self):
        """
        Load variable from file or set to initial value of the variable.
        """
        if self.scale_file is None: return
        value = read_value_json(self.scale_file, self._name)
        if value is None:
            # logging.debug(
            #     f"Initialize variable {self._name}' to {self.variable.numpy():.3f}"
            # )
            pass
        else:
            self._fitted = True
            # logging.debug(f"Set scale factor {self._name} : {value}")
            with torch.no_grad():
                self.variable.copy_(torch.tensor(value))


def real_sph_harm(L_maxdegree, use_theta, use_phi=True, zero_m_only=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to degree L (excluded).
    Variables are either spherical coordinates phi and theta (or cartesian coordinates x,y,z) on the UNIT SPHERE.

    Parameters
    ----------
        L_maxdegree: int
            Degree up to which to calculate the spherical harmonics (degree L is excluded).
        use_theta: bool
            - True: Expects the input of the formula strings to contain theta.
            - False: Expects the input of the formula strings to contain z.
        use_phi: bool
            - True: Expects the input of the formula strings to contain phi.
            - False: Expects the input of the formula strings to contain x and y.
            Does nothing if zero_m_only is True
        zero_m_only: bool
            If True only calculate the harmonics where m=0.

    Returns
    -------
        Y_lm_real: list
            Computes formula strings of the the real part of the spherical harmonics up
            to degree L (where degree L is not excluded).
            In total L^2 many sph harm exist up to degree L (excluded). However, if zero_m_only only is True then
            the total count is reduced to be only L many.
    """
    z = sym.symbols("z")
    P_l_m = associated_legendre_polynomials(L_maxdegree, zero_m_only)
    if zero_m_only:
        # for all m != 0: Y_lm = 0
        Y_l_m = [[0] for l_degree in range(L_maxdegree)]
    else:
        Y_l_m = [
            [0] * (2 * l_degree + 1) for l_degree in range(L_maxdegree)
        ]  # for order l: -l <= m <= l

    # convert expressions to spherical coordiantes
    if use_theta:
        # replace z by cos(theta)
        theta = sym.symbols("theta")
        for l_degree in range(L_maxdegree):
            for m_order in range(len(P_l_m[l_degree])):
                if not isinstance(P_l_m[l_degree][m_order], int):
                    P_l_m[l_degree][m_order] = P_l_m[l_degree][m_order].subs(
                        z, sym.cos(theta)
                    )

    ## calculate Y_lm
    # Y_lm = N * P_lm(cos(theta)) * exp(i*m*phi)
    #             { sqrt(2) * (-1)^m * N * P_l|m| * sin(|m|*phi)   if m < 0
    # Y_lm_real = { Y_lm                                           if m = 0
    #             { sqrt(2) * (-1)^m * N * P_lm * cos(m*phi)       if m > 0

    for l_degree in range(L_maxdegree):
        Y_l_m[l_degree][0] = sym.simplify(
            sph_harm_prefactor(l_degree, 0) * P_l_m[l_degree][0]
        )  # Y_l0

    if not zero_m_only:
        phi = sym.symbols("phi")
        for l_degree in range(1, L_maxdegree):
            # m > 0
            for m_order in range(1, l_degree + 1):
                Y_l_m[l_degree][m_order] = sym.simplify(
                    2 ** 0.5
                    * (-1) ** m_order
                    * sph_harm_prefactor(l_degree, m_order)
                    * P_l_m[l_degree][m_order]
                    * sym.cos(m_order * phi)
                )
            # m < 0
            for m_order in range(1, l_degree + 1):
                Y_l_m[l_degree][-m_order] = sym.simplify(
                    2 ** 0.5
                    * (-1) ** m_order
                    * sph_harm_prefactor(l_degree, -m_order)
                    * P_l_m[l_degree][m_order]
                    * sym.sin(m_order * phi)
                )

        # convert expressions to cartesian coordinates
        if not use_phi:
            # replace phi by atan2(y,x)
            x = sym.symbols("x")
            y = sym.symbols("y")
            for l_degree in range(L_maxdegree):
                for m_order in range(len(Y_l_m[l_degree])):
                    Y_l_m[l_degree][m_order] = sym.simplify(
                        Y_l_m[l_degree][m_order].subs(phi, sym.atan2(y, x))
                    )
    return Y_l_m

def associated_legendre_polynomials(
    L_maxdegree, zero_m_only=True, pos_m_only=True
):
    """Computes string formulas of the associated legendre polynomials up to degree L (excluded).

    Parameters
    ----------
        L_maxdegree: int
            Degree up to which to calculate the associated legendre polynomials (degree L is excluded).
        zero_m_only: bool
            If True only calculate the polynomials for the polynomials where m=0.
        pos_m_only: bool
            If True only calculate the polynomials for the polynomials where m>=0. Overwritten by zero_m_only.

    Returns
    -------
        polynomials: list
            Contains the sympy functions of the polynomials (in total L many if zero_m_only is True else L^2 many).
    """
    # calculations from http://web.cmb.usc.edu/people/alber/Software/tomominer/docs/cpp/group__legendre__polynomials.html
    z = sym.symbols("z")
    P_l_m = [
        [0] * (2 * l_degree + 1) for l_degree in range(L_maxdegree)
    ]  # for order l: -l <= m <= l

    P_l_m[0][0] = 1
    if L_maxdegree > 0:
        if zero_m_only:
            # m = 0
            P_l_m[1][0] = z
            for l_degree in range(2, L_maxdegree):
                P_l_m[l_degree][0] = sym.simplify(
                    (
                        (2 * l_degree - 1) * z * P_l_m[l_degree - 1][0]
                        - (l_degree - 1) * P_l_m[l_degree - 2][0]
                    )
                    / l_degree
                )
            return P_l_m
        else:
            # for m >= 0
            for l_degree in range(1, L_maxdegree):
                P_l_m[l_degree][l_degree] = sym.simplify(
                    (1 - 2 * l_degree)
                    * (1 - z ** 2) ** 0.5
                    * P_l_m[l_degree - 1][l_degree - 1]
                )  # P_00, P_11, P_22, P_33

            for m_order in range(0, L_maxdegree - 1):
                P_l_m[m_order + 1][m_order] = sym.simplify(
                    (2 * m_order + 1) * z * P_l_m[m_order][m_order]
                )  # P_10, P_21, P_32, P_43

            for l_degree in range(2, L_maxdegree):
                for m_order in range(l_degree - 1):  # P_20, P_30, P_31
                    P_l_m[l_degree][m_order] = sym.simplify(
                        (
                            (2 * l_degree - 1)
                            * z
                            * P_l_m[l_degree - 1][m_order]
                            - (l_degree + m_order - 1)
                            * P_l_m[l_degree - 2][m_order]
                        )
                        / (l_degree - m_order)
                    )

            if not pos_m_only:
                # for m < 0: P_l(-m) = (-1)^m * (l-m)!/(l+m)! * P_lm
                for l_degree in range(1, L_maxdegree):
                    for m_order in range(
                        1, l_degree + 1
                    ):  # P_1(-1), P_2(-1) P_2(-2)
                        P_l_m[l_degree][-m_order] = sym.simplify(
                            (-1) ** m_order
                            * np.math.factorial(l_degree - m_order)
                            / np.math.factorial(l_degree + m_order)
                            * P_l_m[l_degree][m_order]
                        )

            return P_l_m

def sph_harm_prefactor(l_degree, m_order):
    """Computes the constant pre-factor for the spherical harmonic of degree l and order m.

    Parameters
    ----------
        l_degree: int
            Degree of the spherical harmonic. l >= 0
        m_order: int
            Order of the spherical harmonic. -l <= m <= l

    Returns
    -------
        factor: float

    """
    # sqrt((2*l+1)/4*pi * (l-m)!/(l+m)! )
    return (
        (2 * l_degree + 1)
        / (4 * np.pi)
        * np.math.factorial(l_degree - abs(m_order))
        / np.math.factorial(l_degree + abs(m_order))
    ) ** 0.5


class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


def _standardize(kernel):
    """
    Makes sure that N*Var(W) = 1 and E[W] = 0
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel

class SiQU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return x * self._activation(x)

def he_orthogonal_init(tensor):
    """
    Generate a weight matrix with variance according to He (Kaiming) initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor

class Dense(torch.nn.Module):
    """
    Combines dense layer with scaling for swish activation.

    Parameters
    ----------
        units: int
            Output embedding size.
        activation: str
            Name of the activation function to use.
        bias: bool
            True if use bias.
    """

    def __init__(self, in_features, out_features, bias=False, activation=None):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation == "siqu":
            self._activation = SiQU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )

    def reset_parameters(self, initializer=he_orthogonal_init):
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        x = self._activation(x)
        return x

class EfficientInteractionDownProjection(torch.nn.Module):
    """
    Down projection in the efficient reformulation.

    Parameters
    ----------
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        emb_size_interm: int,
    ):
        super().__init__()

        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.emb_size_interm = emb_size_interm

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.num_spherical, self.num_radial, self.emb_size_interm)
            ),
            requires_grad=True,
        )
        he_orthogonal_init(self.weight)

    def forward(self, rbf, sph, id_ca, id_ragged_idx):
        """

        Arguments
        ---------
        rbf: torch.Tensor, shape=(1, nEdges, num_radial)
        sph: torch.Tensor, shape=(nEdges, Kmax, num_spherical)
        id_ca
        id_ragged_idx

        Returns
        -------
        rbf_W1: torch.Tensor, shape=(nEdges, emb_size_interm, num_spherical)
        sph: torch.Tensor, shape=(nEdges, Kmax, num_spherical)
            Kmax = maximum number of neighbors of the edges
        """
        num_edges = rbf.shape[1]

        # MatMul: mul + sum over num_radial
        rbf_W1 = torch.matmul(rbf.float(), self.weight)
        # (num_spherical, nEdges , emb_size_interm)
        rbf_W1 = rbf_W1.permute(1, 2, 0)
        # (nEdges, emb_size_interm, num_spherical)

        # Zero padded dense matrix
        # maximum number of neighbors, catch empty id_ca with maximum
        if sph.shape[0] == 0:
            Kmax = 0
        else:
            Kmax = torch.max(
                torch.max(id_ragged_idx + 1),
                torch.tensor(0).to(id_ragged_idx.device),
            )

        sph2 = sph.new_zeros(num_edges, Kmax, self.num_spherical)
        sph2[id_ca, id_ragged_idx] = sph

        sph2 = torch.transpose(sph2, 1, 2)
        # (nEdges, num_spherical/emb_size_interm, Kmax)

        return rbf_W1, sph2


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size
        # Atom embeddings: We go up to Bi (83).
        self.embeddings = torch.nn.Embedding(MAX_ATOMIC_NUM, emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z)
        # h = self.embeddings(Z - 1) # -1 because Z.min()=1 (==Hydrogen)
        return h


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        emb_size: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(
        self,
        atom_features,
        edge_features,
        out_features,
        activation=None,
    ):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(
        self,
        h,
        m_rbf,
        idx_s,
        idx_t,
    ):
        """

        Arguments
        ---------
        h
        m_rbf: shape (nEdges, nFeatures)
            in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_st
        idx_s
        idx_t

        Returns
        -------
            m_st: torch.Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_s = h[idx_s]  # shape=(nEdges, emb_size)
        h_t = h[idx_t]  # shape=(nEdges, emb_size)

        m_st = torch.cat([h_s, h_t, m_rbf], dim=-1)  # (nEdges, 2*emb_size+nFeatures)
        m_st = self.dense(m_st)  # (nEdges, emb_size)
        return m_st


class EfficientInteractionBilinear(torch.nn.Module):
    """
    Efficient reformulation of the bilinear layer and subsequent summation.

    Parameters
    ----------
        units_out: int
            Embedding output size of the bilinear layer.
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(
        self,
        emb_size: int,
        emb_size_interm: int,
        units_out: int,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_interm = emb_size_interm
        self.units_out = units_out

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.emb_size, self.emb_size_interm, self.units_out),
                requires_grad=True,
            )
        )
        he_orthogonal_init(self.weight)

    def forward(
        self,
        basis,
        m,
        id_reduce,
        id_ragged_idx,
    ):
        """

        Arguments
        ---------
        basis
        m: quadruplets: m = m_db , triplets: m = m_ba
        id_reduce
        id_ragged_idx

        Returns
        -------
            m_ca: torch.Tensor, shape=(nEdges, units_out)
                Edge embeddings.
        """
        # num_spherical is actually num_spherical**2 for quadruplets
        (rbf_W1, sph) = basis
        # (nEdges, emb_size_interm, num_spherical), (nEdges, num_spherical, Kmax)
        nEdges = rbf_W1.shape[0]

        # Create (zero-padded) dense matrix of the neighboring edge embeddings.
        Kmax = torch.max(
            torch.max(id_ragged_idx) + 1,
            torch.tensor(0).to(id_ragged_idx.device),
        )
        # maximum number of neighbors, catch empty id_reduce_ji with maximum
        m2 = m.new_zeros(nEdges, Kmax, self.emb_size)
        m2[id_reduce, id_ragged_idx] = m
        # (num_quadruplets or num_triplets, emb_size) -> (nEdges, Kmax, emb_size)

        sum_k = torch.matmul(sph, m2)  # (nEdges, num_spherical, emb_size)

        # MatMul: mul + sum over num_spherical
        rbf_W1_sum_k = torch.matmul(rbf_W1, sum_k)
        # (nEdges, emb_size_interm, emb_size)

        # Bilinear: Sum over emb_size_interm and emb_size
        m_ca = torch.matmul(rbf_W1_sum_k.permute(2, 0, 1), self.weight)
        # (emb_size, nEdges, units_out)
        m_ca = torch.sum(m_ca, dim=0)
        # (nEdges, units_out)

        return m_ca


def update_json(path, data):
    """"""
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    content = read_json(path)
    content.update(data)
    write_json(path, content)


def write_json(path, data):
    """"""
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

class AutoScaleFit(AutomaticFit):
    """
    Class to automatically fit the scaling factors depending on the observed variances.

    Parameters
    ----------
        variable: torch.Tensor
            Variable to fit.
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
    """

    def __init__(self, variable, scale_file, name):
        super().__init__(variable, scale_file, name)

        if not self._fitted:
            self._init_stats()

    def _init_stats(self):
        self.variance_in = 0
        self.variance_out = 0
        self.nSamples = 0

    @torch.no_grad()
    def observe(self, x, y):
        """
        Observe variances for input x and output y.
        The scaling factor alpha is calculated s.t. Var(alpha * y) ~ Var(x)
        """
        if self._fitted:
            return

        # only track stats for current variable
        if AutomaticFit.activeVar == self:
            nSamples = y.shape[0]
            self.variance_in += (
                torch.mean(torch.var(x, dim=0)).to(dtype=torch.float32)
                * nSamples
            )
            self.variance_out += (
                torch.mean(torch.var(y, dim=0)).to(dtype=torch.float32)
                * nSamples
            )
            self.nSamples += nSamples

    @torch.no_grad()
    def fit(self):
        """
        Fit the scaling factor based on the observed variances.
        """
        if AutomaticFit.activeVar == self:
            if self.variance_in == 0:
                raise ValueError(
                    f"Did not track the variable {self._name}. Add observe calls to track the variance before and after."
                )

            # calculate variance preserving scaling factor
            self.variance_in = self.variance_in / self.nSamples
            self.variance_out = self.variance_out / self.nSamples

            ratio = self.variance_out / self.variance_in
            value = torch.sqrt(1 / ratio)
            # logging.info(
            #     f"Variable: {self._name}, "
            #     f"Var_in: {self.variance_in.item():.3f}, "
            #     f"Var_out: {self.variance_out.item():.3f}, "
            #     f"Ratio: {ratio:.3f} => Scaling factor: {value:.3f}"
            # )

            # set variable to calculated value
            self.variable.copy_(self.variable * value)
            update_json(
                self.scale_file, {self._name: float(self.variable.item())}
            )
            self.set_next_active()  # set next variable in queue to active

class ScalingFactor(torch.nn.Module):
    """
    Scale the output y of the layer s.t. the (mean) variance wrt. to the reference input x_ref is preserved.

    Parameters
    ----------
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
        name: str
            Name of the scaling factor
    """

    def __init__(self, scale_file, name, device=None):
        super().__init__()

        self.scale_factor = torch.nn.Parameter(
            torch.tensor(1.0, device=device), requires_grad=False
        )
        self.autofit = AutoScaleFit(self.scale_factor, scale_file, name)

    def forward(self, x_ref, y):
        y = y * self.scale_factor
        self.autofit.observe(x_ref, y)

        return y

class TripletInteraction(torch.nn.Module):
    """
    Triplet-based message passing block.

    Parameters
    ----------
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size of the edge embeddings after the hadamard product with rbf.
        emb_size_bilinear: int
            Embedding size of the edge embeddings after the bilinear layer.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).

        activation: str
            Name of the activation function to use in the dense layers except for the final dense layer.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_edge,
        emb_size_trip,
        emb_size_bilinear,
        emb_size_rbf,
        emb_size_cbf,
        activation=None,
        scale_file=None,
        name="TripletInteraction",
        **kwargs,
    ):
        super().__init__()
        self.name = name

        # Dense transformation
        self.dense_ba = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        # Up projections of basis representations, bilinear layer and scaling factors
        self.mlp_rbf = Dense(
            emb_size_rbf,
            emb_size_edge,
            activation=None,
            bias=False,
        )
        self.scale_rbf = ScalingFactor(
            scale_file=scale_file, name=name + "_had_rbf"
        )

        self.mlp_cbf = EfficientInteractionBilinear(
            emb_size_trip, emb_size_cbf, emb_size_bilinear
        )
        self.scale_cbf_sum = ScalingFactor(
            scale_file=scale_file, name=name + "_sum_cbf"
        )  # combines scaling for bilinear layer and summation

        # Down and up projections
        self.down_projection = Dense(
            emb_size_edge,
            emb_size_trip,
            activation=activation,
            bias=False,
        )
        self.up_projection_ca = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
        )
        self.up_projection_ac = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(
        self,
        m,
        rbf3,
        cbf3,
        id3_ragged_idx,
        id_swap,
        id3_ba,
        id3_ca,
    ):
        """
        Returns
        -------
            m: torch.Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """

        # Dense transformation
        x_ba = self.dense_ba(m)  # (nEdges, emb_size_edge)

        # Transform via radial bessel basis
        rbf_emb = self.mlp_rbf(rbf3)  # (nEdges, emb_size_edge)
        x_ba2 = x_ba * rbf_emb
        x_ba = self.scale_rbf(x_ba, x_ba2)

        x_ba = self.down_projection(x_ba)  # (nEdges, emb_size_trip)

        # Transform via circular spherical basis
        x_ba = x_ba[id3_ba]

        # Efficient bilinear layer
        x = self.mlp_cbf(cbf3, x_ba, id3_ca, id3_ragged_idx)
        # (nEdges, emb_size_quad)
        x = self.scale_cbf_sum(x_ba, x)

        # =>
        # rbf(d_ba)
        # cbf(d_ca, angle_cab)

        # Up project embeddings
        x_ca = self.up_projection_ca(x)  # (nEdges, emb_size_edge)
        x_ac = self.up_projection_ac(x)  # (nEdges, emb_size_edge)

        # Merge interaction of c->a and a->c
        x_ac = x_ac[id_swap]  # swap to add to edge a->c and not c->a
        x3 = x_ca + x_ac
        x3 = x3 * self.inv_sqrt_2
        return x3

class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        if type(act) == str and act == 'swish':
            self.act = ScaledSiLU()
        else: self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.lin2(self.act(self.lin1(x))))

class InteractionBlockTripletsOnly(torch.nn.Module):
    """
    Interaction block for GemNet-T/dT.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).

        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.

        activation: str
            Name of the activation function to use in the dense layers except for the final dense layer.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom,
        emb_size_edge,
        emb_size_trip,
        emb_size_rbf,
        emb_size_cbf,
        emb_size_bil_trip,
        num_before_skip,
        num_after_skip,
        num_concat,
        num_atom,
        activation=None,
        scale_file=None,
        name="Interaction",
    ):
        super().__init__()
        self.name = name

        block_nr = name.split("_")[-1]

        ## -------------------------------------------- Message Passing ------------------------------------------- ##
        # Dense transformation of skip connection
        self.dense_ca = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        # Triplet Interaction
        self.trip_interaction = TripletInteraction(
            emb_size_edge=emb_size_edge,
            emb_size_trip=emb_size_trip,
            emb_size_bilinear=emb_size_bil_trip,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            activation=activation,
            scale_file=scale_file,
            name=f"TripInteraction_{block_nr}",
        )

        ## ---------------------------------------- Update Edge Embeddings ---------------------------------------- ##
        # Residual layers before skip connection
        self.layers_before_skip = torch.nn.ModuleList(
            [
                ResidualLayer(
                    emb_size_edge,
                    act=activation,
                )
                for i in range(num_before_skip)
            ]
        )

        # Residual layers after skip connection
        self.layers_after_skip = torch.nn.ModuleList(
            [
                ResidualLayer(
                    emb_size_edge,
                    act=activation,
                )
                for i in range(num_after_skip)
            ]
        )

        ## ---------------------------------------- Update Atom Embeddings ---------------------------------------- ##
        self.atom_update = AtomUpdateBlock(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=num_atom,
            activation=activation,
            scale_file=scale_file,
            name=f"AtomUpdate_{block_nr}",
        )

        ## ------------------------------ Update Edge Embeddings with Atom Embeddings ----------------------------- ##
        self.concat_layer = EdgeEmbedding(
            emb_size_atom,
            emb_size_edge,
            emb_size_edge,
            activation=activation,
        )
        self.residual_m = torch.nn.ModuleList(
            [
                ResidualLayer(emb_size_edge, act=activation)
                for _ in range(num_concat)
            ]
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(
        self,
        h,
        m,
        rbf3,
        cbf3,
        id3_ragged_idx,
        id_swap,
        id3_ba,
        id3_ca,
        rbf_h,
        idx_s,
        idx_t,
    ):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nEdges, emb_size_atom)
                Atom embeddings.
            m: torch.Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """

        # Initial transformation
        x_ca_skip = self.dense_ca(m)  # (nEdges, emb_size_edge)

        x3 = self.trip_interaction(
            m,
            rbf3,
            cbf3,
            id3_ragged_idx,
            id_swap,
            id3_ba,
            id3_ca,
        )

        ## ----------------------------- Merge Embeddings after Triplet Interaction ------------------------------ ##
        x = x_ca_skip + x3  # (nEdges, emb_size_edge)
        x = x * self.inv_sqrt_2

        ## ---------------------------------------- Update Edge Embeddings --------------------------------------- ##
        # Transformations before skip connection
        for i, layer in enumerate(self.layers_before_skip):
            x = layer(x)  # (nEdges, emb_size_edge)

        # Skip connection
        m = m + x  # (nEdges, emb_size_edge)
        m = m * self.inv_sqrt_2

        # Transformations after skip connection
        for i, layer in enumerate(self.layers_after_skip):
            m = layer(m)  # (nEdges, emb_size_edge)

        ## ---------------------------------------- Update Atom Embeddings --------------------------------------- ##
        h2 = self.atom_update(h, m, rbf_h, idx_t)

        # Skip connection
        h = h + h2  # (nAtoms, emb_size_atom)
        h = h * self.inv_sqrt_2

        ## ----------------------------- Update Edge Embeddings with Atom Embeddings ----------------------------- ##
        m2 = self.concat_layer(h, m, idx_s, idx_t)  # (nEdges, emb_size_edge)

        for i, layer in enumerate(self.residual_m):
            m2 = layer(m2)  # (nEdges, emb_size_edge)

        # Skip connection
        m = m + m2  # (nEdges, emb_size_edge)
        m = m * self.inv_sqrt_2
        return h, m


class AtomUpdateBlock(torch.nn.Module):
    """
    Aggregate the message embeddings of the atoms

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_atom: int
            Embedding size of the edges.
        nHidden: int
            Number of residual blocks.
        activation: callable/str
            Name of the activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        activation=None,
        scale_file=None,
        name: str = "atom_update",
    ):
        super().__init__()
        self.name = name

        self.dense_rbf = Dense(
            emb_size_rbf, emb_size_edge, activation=None, bias=False
        )
        self.scale_sum = ScalingFactor(
            scale_file=scale_file, name=name + "_sum"
        )

        self.layers = self.get_mlp(
            emb_size_edge, emb_size_atom, nHidden, activation
        )

    def get_mlp(self, units_in, units, nHidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            # ResidualLayer(units, nLayers=2, act=activation)
            ResidualLayer(units, act=activation)
            for i in range(nHidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(self, h, m, rbf, id_j):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size_atom)
                Atom embedding.
        """
        nAtoms = h.shape[0]

        mlp_rbf = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * mlp_rbf

        x2 = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce="sum")
        # (nAtoms, emb_size_edge)
        x = self.scale_sum(m, x2)

        for layer in self.layers:
            x = layer(x)  # (nAtoms, emb_size_atom)

        return x

class OutputBlock(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_atom: int
            Embedding size of the edges.
        nHidden: int
            Number of residual blocks.
        num_targets: int
            Number of targets.
        activation: str
            Name of the activation function to use in the dense layers except for the final dense layer.
        direct_forces: bool
            If true directly predict forces without taking the gradient of the energy potential.
        output_init: int
            Kernel initializer of the final dense layer.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        num_targets: int,
        activation=None,
        direct_forces=True,
        output_init="HeOrthogonal",
        scale_file=None,
        name: str = "output",
        **kwargs,
    ):

        super().__init__(
            name=name,
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
            scale_file=scale_file,
            **kwargs,
        )

        assert isinstance(output_init, str)
        self.output_init = output_init.lower()
        self.direct_forces = direct_forces

        self.seq_energy = self.layers  # inherited from parent class
        self.out_energy = Dense(
            emb_size_atom, num_targets, bias=False, activation=None
        )

        if self.direct_forces:
            self.scale_rbf_F = ScalingFactor(
                scale_file=scale_file, name=name + "_had"
            )
            self.seq_forces = self.get_mlp(
                emb_size_edge, emb_size_edge, nHidden, activation
            )
            self.out_forces = Dense(
                emb_size_edge, num_targets, bias=False, activation=None
            )
            self.dense_rbf_F = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )

        self.reset_parameters()

    def reset_parameters(self):
        if self.output_init == "heorthogonal":
            self.out_energy.reset_parameters(he_orthogonal_init)
            if self.direct_forces:
                self.out_forces.reset_parameters(he_orthogonal_init)
        elif self.output_init == "zeros":
            self.out_energy.reset_parameters(torch.nn.init.zeros_)
            if self.direct_forces:
                self.out_forces.reset_parameters(torch.nn.init.zeros_)
        else:
            raise UserWarning(f"Unknown output_init: {self.output_init}")

    def forward(self, h, m, rbf, id_j):
        """
        Returns
        -------
            (E, F): tuple
            - E: torch.Tensor, shape=(nAtoms, num_targets)
            - F: torch.Tensor, shape=(nEdges, num_targets)
            Energy and force prediction
        """
        nAtoms = h.shape[0]

        # -------------------------------------- Energy Prediction -------------------------------------- #
        rbf_emb_E = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * rbf_emb_E

        x_E = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce="sum")
        # (nAtoms, emb_size_edge)
        x_E = self.scale_sum(m, x_E)

        for layer in self.seq_energy:
            x_E = layer(x_E)  # (nAtoms, emb_size_atom)

        x_E = self.out_energy(x_E)  # (nAtoms, num_targets)

        # --------------------------------------- Force Prediction -------------------------------------- #
        if self.direct_forces:
            x_F = m
            for i, layer in enumerate(self.seq_forces):
                x_F = layer(x_F)  # (nEdges, emb_size_edge)

            rbf_emb_F = self.dense_rbf_F(rbf)  # (nEdges, emb_size_edge)
            x_F_rbf = x_F * rbf_emb_F
            x_F = self.scale_rbf_F(x_F, x_F_rbf)

            x_F = self.out_forces(x_F)  # (nEdges, num_targets)
        else:
            x_F = 0
        # ----------------------------------------------------------------------------------------------- #

        return x_E, x_F


def ragged_range(sizes):
    """Multiple concatenated ranges.

    Examples
    --------
        sizes = [1 4 2 3]
        Return: [0  0 1 2 3  0 1  0 1 2]
    """
    assert sizes.dim() == 1
    if sizes.sum() == 0:
        return sizes.new_empty(0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        sizes = torch.masked_select(sizes, sizes_nonzero)

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    id_steps = torch.ones(sizes.sum(), dtype=torch.long, device=sizes.device)
    id_steps[0] = 0
    insert_index = sizes[:-1].cumsum(0)
    insert_val = (1 - sizes)[:-1]

    # Assign index-offsetting values
    id_steps[insert_index] = insert_val

    # Finally index into input array for the group repeated o/p
    res = id_steps.cumsum(0)
    return res


def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing=True,
    start_idx=0,
    block_inc=0,
    repeat_inc=0,
):
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(
        torch.arange(len(sizes), device=sizes.device), repeats
    )

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(
                block_inc[: r1[-1]], indptr, reduce="sum"
            )
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res


def inner_product_normalized(x, y):
    """
    Calculate the inner product between the given normalized vectors,
    giving a result between -1 and 1.
    """
    return torch.sum(x * y, dim=-1).clamp(min=-1, max=1)


def mask_neighbors(neighbors, edge_mask):
    neighbors_old_indptr = torch.cat([neighbors.new_zeros(1), neighbors])
    neighbors_old_indptr = torch.cumsum(neighbors_old_indptr, dim=0)
    neighbors = segment_csr(edge_mask.long(), neighbors_old_indptr)
    return neighbors

def sparse_to_dense_coords(sparse, batch_size, tot_atoms, n_atoms):
    dense = torch.empty(size=(tot_atoms, 3), dtype=sparse.dtype, device=sparse.device)
    pos = 0
    for i in range(batch_size):
        dense[pos:pos+n_atoms[i]] = sparse[i,0:n_atoms[i]]
        pos += n_atoms[i]
    return dense

def dense_to_sparse_coords(dense, batch_size, max_atoms, n_atoms):
    sparse = torch.zeros((batch_size, max_atoms, 3), dtype=dense.dtype, device=dense.device)
    pos = 0
    for i in range(batch_size):
        sparse[i,0:n_atoms[i]] = dense[pos:pos+n_atoms[i]]
        pos += n_atoms[i]
    return sparse

def cif2png(fname, ofname):
    atoms = read(fname)
    write(ofname, atoms, rotation="45y, 15x")