import numpy as np
import networkx as nx
from scipy import linalg
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import pickle

class EchoStateNetwork:
    def __init__(self, 
                 n_reservoir=300,
                 spectral_radius=0.85,
                 input_scaling=0.5,
                 leak_rate=0.5,
                 connectivity=0.75,
                 regularization=1e-6,
                 n_warmup=40001,
                 bias=0.0,
                 random_state=None):
        """
        Echo State Network implementation
        
        Parameters:
        -----------
        n_reservoir : int (300-1000), 
            Number of reservoir neurons
        spectral_radius : float (0, 1.5]
            Spectral radius of reservoir matrix
        input_scaling : float
            Scaling factor for input weights (Win range: [-0.5, 0.5])
        leak_rate : float (0, 1]
            Leaky coefficient alpha
        connectivity : float (0, 1]
            Connection probability p for Erdős-Rényi graph
        regularization : float
            Ridge regression regularization parameter lambda
        n_warmup : int (>= 40001)
            Number of warmup steps
        bias : float
            Bias term b
        random_state : int, optional
            Random seed for reproducibility
        """

        # Store hyperparameters
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.connectivity = connectivity
        self.regularization = regularization
        self.n_warmup = n_warmup
        self.bias = bias
        
        # Set random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize matrices
        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.state = None
        
    def _create_reservoir_matrix(self):
        """Create reservoir weight matrix using Erdős-Rényi random graph, which is
        then used to scale the resulting matrix to achieve the desired spectral radius."""

        # Create Erdős-Rényi graph with self-loops
        # A random topology where each possible edge exists with probability p
        G = nx.erdos_renyi_graph(self.n_reservoir, self.connectivity, directed=True, seed=None)
        
        # Add self-loops with probability self.connectivity
        for node in G.nodes():
            if np.random.random() < self.connectivity:
                G.add_edge(node, node)
        
        # Convert to adjacency matrix
        A = nx.adjacency_matrix(G, dtype=float).toarray()
        
        # Create random weights for all possible connections
        W_res = np.random.uniform(-1, 1, (self.n_reservoir, self.n_reservoir))

        # Element-wise multiplication preserves topology
        W_res = W_res * A  
        
        # Compute eigenvalues to find current spectral radius
        eigenvalues = np.linalg.eigvals(W_res)
        current_spectral_radius = np.max(np.abs(eigenvalues))
        
        # Scale weights to achieve desired spectral radius
        if current_spectral_radius > 0:
            W_res = W_res * (self.spectral_radius / current_spectral_radius)
        
        return W_res
    
    def _create_input_matrix(self, n_inputs):
        """Create input weight matrix Win"""
        W_in = np.random.uniform(-self.input_scaling, self.input_scaling, 
                                (self.n_reservoir, n_inputs))
        return W_in
    
    def _update_state(self, input_vector, state):
        """Update reservoir state"""
        # Compute new state: (1-α) * x(t) + α * tanh(W_res * x(t) + W_in * u(t) + b)
        new_state = (1 - self.leak_rate) * state + \
                    self.leak_rate * np.tanh(
                        np.dot(self.W_res, state) + 
                        np.dot(self.W_in, input_vector) + 
                        self.bias
                    )
        return new_state
    
    def fit(self, X, y):
        """
        Train the ESN
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input time series data
        y : array-like, shape (n_samples, n_outputs)
            Target output data
        """
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        n_samples, n_inputs = X.shape
        # n_outputs = y.shape[1]
        
        # Initialize matrices if not done already
        if self.W_in is None:
            self.W_in = self._create_input_matrix(n_inputs) # size: n_reservoir x n_inputs
        if self.W_res is None:
            self.W_res = self._create_reservoir_matrix() # size: n_reservoir x n_reservoir

        # Initialize state 
        self.state = np.zeros(self.n_reservoir)
        
        # Collect reservoir states
        states = []

        # Update reservoir states with current input
        for i in range(n_samples):
            self.state = self._update_state(X[i], self.state)

            # Store the state for training later
            states.append(self.state.copy())

        # Convert to array
        states = np.array(states)
        
        # Warmup: discard initial transient states
        warmup_end = min(self.n_warmup, n_samples)
        states_train = states[warmup_end:]
        y_train = y[warmup_end:]
        
        if len(states_train) == 0:
            raise ValueError("Not enough data points after warmup period")
        
        # Train output weights using Ridge regression
        self.ridge = Ridge(alpha=self.regularization, fit_intercept=False)
        self.ridge.fit(states_train, y_train)

        # Extracted trained output weights
        self.W_out = self.ridge.coef_.T

        return self # returns the model itself for predict()

    def predict(self, X):
        """
        Predict using the trained ESN
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input time series data
            
        Returns:
        --------
        y_pred : array, shape (n_samples, n_outputs)
            Predicted outputs
        """
        if self.W_out is None:
            raise ValueError("Model must be trained before prediction")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        
        # Reset state
        self.state = np.zeros(self.n_reservoir)
        
        # Collect predictions
        predictions = []
        
        for i in range(n_samples):
            self.state = self._update_state(X[i], self.state)
            prediction = np.dot(self.W_out.T, self.state)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    # note: necessary?
    def get_reservoir_states(self, X):
        """
        Get reservoir states for input sequence
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input time series data
            
        Returns:
        --------
        states : array, shape (n_samples, n_reservoir)
            Reservoir states
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]

        # Reset reservoir state
        self.state = np.zeros(self.n_reservoir)
        
        # Collect states
        states = []
        
        for i in range(n_samples):
            self.state = self._update_state(X[i], self.state)
            states.append(self.state.copy())
        
        return np.array(states)

# note: need to remove or modify
def generate_sample_time_series():
    """Generate sample time series for different problem types"""
    np.random.seed(42)
    t = np.linspace(0, 50, 5000)
    
    # Problem A2(a): Simple sinusoidal
    problem_a = np.sin(t) + 0.3 * np.sin(3 * t)
    
    # Problem A2(b): Chaotic system (Lorenz-like)
    dt = 0.01
    n_steps = len(t)
    x = np.zeros(n_steps)
    x[0] = 1.0
    for i in range(1, n_steps):
        x[i] = x[i-1] + dt * (10 * (np.sin(t[i-1]) - x[i-1]))
    problem_b = x + 0.1 * np.random.randn(len(x))
    
    # Problem A2(c): Multi-frequency signal
    problem_c = (np.sin(t) + 0.5 * np.sin(2.5 * t) + 
                0.3 * np.sin(7 * t) + 0.2 * np.cos(0.5 * t))
    
    # Problem A2(e): Nonlinear autoregressive
    problem_e = np.zeros(len(t))
    problem_e[0] = 0.1
    for i in range(1, len(t)):
        problem_e[i] = 0.8 * problem_e[i-1] - 0.5 * problem_e[i-1]**2 + np.sin(t[i]) * 0.3
    
    return {
        'Problem A2(a)': problem_a,
        'Problem A2(b)': problem_b,
        'Problem A2(c)': problem_c,
        'Problem A2(e)': problem_e
    }

def calculate_nrmse(y_true, y_pred):
    """Calculate Normalized Root Mean Square Error"""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    y_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / y_range if y_range != 0 else rmse
    return nrmse

def hyperparameter_analysis_plot(problems_dict, n_trials=3):
    """
    Analyze ESN performance across different hyperparameters
    
    Parameters:
    -----------
    problems_dict : dict
        Dictionary with problem names as keys and time series as values
    n_trials : int
        Number of trials to average over for each parameter setting
    """    
    # Define hyperparameter ranges
    ten = 10 # 10 equidistant points including boundary values
    hyperparams = {
        'n_reservoir': np.linspace(300, 1000, ten, dtype=int), # N_res
        'connectivity': np.linspace(0.1, 1.0, ten),            # p
        'leak_rate': np.linspace(0.1, 1.0, ten),               # alpha
        'spectral_radius': np.linspace(0.1, 1.5, ten)          # rho
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle('NRMSE Training Performance with 4 Hyperparameters', fontsize=16)
    
    for prob_idx, (problem_name, time_series) in enumerate(problems_dict.items()):
        print(f"Analyzing {problem_name}")
        
        # Prepare data
        X = time_series[:-1].reshape(-1, 1)
        y = time_series[1:].reshape(-1, 1)
        
        # Use first 50% for training
        n_train = int(0.5 * len(X))
        X_train, y_train = X[:n_train], y[:n_train]
        
        for param_idx, (param_name, param_values) in enumerate(hyperparams.items()):
            ax = axes[prob_idx, param_idx]
            nrmse_values = []
            
            for param_value in param_values:
                trial_nrmse = []
                
                for trial in range(n_trials):
                    try:
                        # Set up parameters
                        kwargs = {
                            'n_reservoir': 300,
                            'spectral_radius': 0.85,
                            'leak_rate': 0.5,
                            'connectivity': 0.75,
                            'regularization': 1e-6,
                            'n_warmup': 40001,
                            'random_state': 42 + trial
                        }
                        kwargs[param_name] = param_value
                        
                        # Train ESN
                        esn = EchoStateNetwork(**kwargs)
                        esn.fit(X_train, y_train)
                        
                        # Get training predictions (after warmup)
                        warmup_end = min(kwargs['n_warmup'], len(X_train))
                        X_train_eval = X_train[warmup_end:]
                        y_train_eval = y_train[warmup_end:]
                        
                        if len(X_train_eval) > 0:
                            # Reset state for clean prediction
                            esn.state = np.zeros(esn.n_reservoir)
                            
                            # Run through warmup again
                            for i in range(warmup_end):
                                esn.state = esn._update_state(X_train[i], esn.state)
                            
                            # Get predictions for evaluation period
                            y_pred_eval = []
                            for i in range(len(X_train_eval)):
                                esn.state = esn._update_state(X_train_eval[i], esn.state)
                                pred = np.dot(esn.W_out.T, esn.state)
                                y_pred_eval.append(pred)
                            
                            y_pred_eval = np.array(y_pred_eval)
                            nrmse = calculate_nrmse(y_train_eval, y_pred_eval)
                            trial_nrmse.append(nrmse)
                        
                    except Exception as e:
                        print(f"Error with {param_name}={param_value}, trial {trial}: {e}")
                        continue
                
                # if the list is not empty
                if trial_nrmse:
                    nrmse_values.append(np.mean(trial_nrmse))
                else:
                    nrmse_values.append(np.nan)
            
            # Plot results
            ax.plot(param_values, nrmse_values, 'b-o', markersize=4, linewidth=1.5)
            ax.set_xlabel(param_name.replace('_', ' ').title())
            ax.set_ylabel('NRMSE_T')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{problem_name}')
            
            # Set reasonable y-limits
            valid_values = [v for v in nrmse_values if not np.isnan(v)]
            if valid_values:
                y_min, y_max = min(valid_values), max(valid_values)
                y_range = y_max - y_min
                ax.set_ylim(max(0, y_min - 0.1*y_range), y_max + 0.1*y_range)
    
    plt.tight_layout()
    plt.show()
    
    return fig


# Example usage and hyperparameter analysis
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
    
#     # Generate sample problems
#     problems = generate_sample_time_series()
    
#     # Run hyperparameter analysis
#     print("Running hyperparameter analysis...")
#     fig = hyperparameter_analysis_plot(problems, n_trials=2)
    
#     # Also show sample time series
#     fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
#     fig2.suptitle('Sample Time Series Data', fontsize=14)
    
#     for idx, (name, series) in enumerate(problems.items()):
#         ax = axes[idx//2, idx%2]
#         ax.plot(series[:1000], linewidth=1)  # Plot first 1000 points
#         ax.set_title(name)
#         ax.grid(True, alpha=0.3)
#         ax.set_xlabel('Time steps')
#         ax.set_ylabel('Value')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Simple demonstration
#     print("\nSimple ESN demonstration:")
#     # Use Problem A2(a) for demonstration
#     signal = problems['Problem A2(a)']
#     X = signal[:-1].reshape(-1, 1)
#     y = signal[1:].reshape(-1, 1)
    
#     n_train = int(0.7 * len(X))
#     X_train, X_test = X[:n_train], X[n_train:]
#     y_train, y_test = y[:n_train], y[n_train:]
    
#     esn = EchoStateNetwork(n_warmup=200, random_state=42)
#     esn.fit(X_train, y_train)
#     y_pred = esn.predict(X_test)
    
#     nrmse = calculate_nrmse(y_test, y_pred)
#     print(f"Test NRMSE: {nrmse:.6f}")
    
#     # Plot prediction results
#     plt.figure(figsize=(12, 6))
#     plt.plot(y_test[:300], label='True', alpha=0.8)
#     plt.plot(y_pred[:300], label='Predicted', alpha=0.8)
#     plt.legend()
#     plt.title('ESN Prediction Results - Problem A2(a)')
#     plt.xlabel('Time steps')
#     plt.ylabel('Value')
#     plt.grid(True, alpha=0.3)
#     plt.show()