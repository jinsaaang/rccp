from scipy import sparse
import torch
import numpy as np


class Reservoir(object):
    r"""
    Build a reservoir and compute the sequence of the internal states.

    Parameters:
    ------------
    n_internal_units : int (default ``100``)
        Processing units in the reservoir.
    spectral_radius : float (default ``0.99``)
        Largest eigenvalue of the reservoir matrix of connection weights.
        To ensure the Echo State Property, set ``spectral_radius <= leak <= 1``)
    leak : float (default ``None``)
        Amount of leakage in the reservoir state update.
        If ``None`` or ``1.0``, no leakage is used.
    connectivity : float (default ``0.3``)
        Percentage of nonzero connection weights.
        Unused in circle reservoir.
    input_scaling : float (default ``0.2``)
        Scaling of the input connection weights.
        Note that the input weights are randomly drawn from ``{-1,1}``.
    noise_level : float (default ``0.0``)
        Standard deviation of the Gaussian noise injected in the state update.
    circle : bool (default ``False``)
        Generate determinisitc reservoir with circle topology where each connection
        has the same weight.
    """

    def __init__(
        self,
        n_internal_units=100,
        spectral_radius=0.99,
        leak=None,
        connectivity=0.3,
        input_scaling=0.2,
        noise_level=0.0,
        circle=False,
        seed=0,
    ):

        # Initialize hyperparameters
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._noise_level = noise_level
        self._leak = leak
        self._seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Reservoir seed: {self._seed}")

        # Input weights depend on input size: they are set when data is provided
        self._input_weights = None

        # Generate internal weights
        if circle:
            self._internal_weights = self._initialize_internal_weights_Circ(
                n_internal_units, spectral_radius
            )
        else:
            self._internal_weights = self._initialize_internal_weights(
                n_internal_units, connectivity, spectral_radius
            )

    def _initialize_internal_weights_Circ(self, n_internal_units, spectral_radius):
        """Generate internal weights with circular topology."""

        # Construct reservoir with circular topology
        internal_weights = np.zeros(
            (n_internal_units, n_internal_units),
        )
        internal_weights[0, -1] = 1.0
        for i in range(n_internal_units - 1):
            internal_weights[i + 1, i] = 1.0

        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max) / spectral_radius

        return torch.tensor(internal_weights, dtype=torch.float32)

    def _initialize_internal_weights(
        self, n_internal_units, connectivity, spectral_radius
    ):
        """Generate internal weights with a sparse, uniformly random topology."""

        # Generate sparse, uniformly distributed weights.
        internal_weights = sparse.rand(
            n_internal_units, n_internal_units, density=connectivity
        ).todense()

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        internal_weights[np.where(internal_weights > 0)] -= 0.5

        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max) / spectral_radius

        return torch.tensor(internal_weights, dtype=torch.float32)

    def _compute_state_matrix(self, X, n_drop=0, previous_state=None):
        """Compute the reservoir states on input data X."""

        N, T, _ = X.shape
        if previous_state is None:
            previous_state = torch.zeros(
                (N, self._n_internal_units),
                device=X.device,
                dtype=torch.float32,
            )

        # Storage
        if T - n_drop > 0:
            window_size = T - n_drop
        else:
            window_size = T

        state_matrix = torch.empty(
            (N, window_size, self._n_internal_units),
            device=X.device,
            dtype=torch.float32,
        )

        for t in range(T):
            current_input = X[:, t, :]

            # Calculate state
            state_before_tanh = torch.matmul(
                self._internal_weights, previous_state.T
            ) + torch.matmul(self._input_weights, current_input.T)

            # Add noise
            state_before_tanh += (
                torch.rand(
                    self._n_internal_units,
                    N,
                    device=X.device,
                )
                * self._noise_level
            )

            # Apply nonlinearity and leakage (optional)
            # if self._leak is None:
            #     previous_state = np.tanh(state_before_tanh).T
            # else:
            #     previous_state = (1.0 - self._leak) * previous_state + np.tanh(
            #         state_before_tanh
            #     ).T
            if self._leak is None:
                previous_state = torch.tanh(state_before_tanh).T
            else:
                previous_state = (1.0 - self._leak) * previous_state + torch.tanh(
                    state_before_tanh
                ).T

            # Store everything after the dropout period
            if T - n_drop > 0 and t > n_drop - 1:
                state_matrix[:, t - n_drop, :] = previous_state
            elif T - n_drop <= 0:
                state_matrix[:, t, :] = previous_state

        return state_matrix

    def get_states(self, X, n_drop=0, bidir=True, initial_state=None):
        r"""
        Compute reservoir states and return them.

        Parameters:
        ------------
        X : np.ndarray
            Time series, 3D array of shape ``[N,T,V]``, where ``N`` is the number of time series,
            ``T`` is the length of each time series, and ``V`` is the number of variables in each
            time point.
        n_drop : int (default is ``0``)
            Washout period, i.e., number of initial samples to drop due to the transient phase.
        bidir : bool (default is ``True``)
            If ``True``, use bidirectional reservoir
        initial_state : np.ndarray (default is ``None``)
            Initialize the first state of the reservoir to the given value.
            If ``None``, the initial states is a zero-vector.

        Returns:
        ------------
        states : np.ndarray
            Reservoir states, 3D array of shape ``[N,T,n_internal_units]``, where ``N`` is the number
            of time series, ``T`` is the length of each time series, and ``n_internal_units`` is the
            number of processing units in the reservoir.
        """

        N, T, V = X.shape
        if self._input_weights is None:
            self._input_weights = (
                2.0 * np.random.binomial(1, 0.5, [self._n_internal_units, V]) - 1.0
            ) * self._input_scaling
            self._input_weights = torch.tensor(
                self._input_weights, device=X.device, dtype=torch.float32
            )

        # Compute sequence of reservoir states
        states = self._compute_state_matrix(X, n_drop, previous_state=initial_state)

        # Reservoir states on time reversed input
        if bidir is True:
            X_r = X[:, ::-1, :]
            states_r = self._compute_state_matrix(X_r, n_drop)
            states = torch.cat((states, states_r), axis=2)

        return states

    def to(self, device):
        """Move the reservoir to the specified device."""
        if self._internal_weights is not None:
            self._internal_weights = self._internal_weights.to(device)
        if self._input_weights is not None:
            self._input_weights = self._input_weights.to(device)
        # if self._n_internal_units is not None:
        #     self._n_internal_units = self._n_internal_units.to(device)
        # if self._leak is not None:
        #     self._leak = self._leak.to(device)
        # if self._input_scaling is not None:
        #     self._input_scaling = self._input_scaling.to(device)
        # if self._noise_level is not None:
        #     self._noise_level = self._noise_level.to(device)

        return self
