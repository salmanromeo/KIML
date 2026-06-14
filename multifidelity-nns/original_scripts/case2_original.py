import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import tensorflow as tf
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ======================
# NATURE-STYLE PLOTTING
# ======================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.fontsize'] = 9

# ======================
# COMMON FUNCTIONS & PHYSICS (TENSORFLOW COMPATIBLE)
# ======================
def y_HF(t):
    if isinstance(t, tf.Tensor):
        return tf.sin(20.0*t) + t + 0.2*tf.sin(20.0*t)*tf.cos(10.0*t)
    else:
        return np.sin(20*t) + t + 0.2*np.sin(20*t)*np.cos(10*t)

def y_LF(t):
    if isinstance(t, tf.Tensor):
        return (0.8*tf.sin(18.0*t) + 0.6*t + 0.4
                + 0.15*tf.sin(18.0*t)*tf.sin(8.0*t)
                + 0.1*t*tf.cos(12.0*t))
    else:
        return (0.8*np.sin(18*t) + 0.6*t + 0.4
                + 0.15*np.sin(18*t)*np.sin(8*t)
                + 0.1*t*np.cos(12*t))

# TensorFlow version for physics residual
def physics_residual_tf(t, y_func):
    """Compute physics residual for TensorFlow models: y'' + 400y - 400t"""
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t)
        with tf.GradientTape() as tape1:
            tape1.watch(t)
            y_val = y_func(t)
        dy_dt = tape1.gradient(y_val, t)
    d2y_dt2 = tape2.gradient(dy_dt, t)

    # Clean up the persistent tape
    del tape2

    return d2y_dt2 + 400.0 * y_val - 400.0 * t

# ======================
# CASE 3: STANDARD NN (WITH PHYSICS)
# ======================
def standard_NN(N_HF, N_LF, physics_weight=0.5, physics_std=10.0, plot_results=True):
    """
    Physics-Informed Neural Network with Physics Loss
    Case: Linear Scaling
    """
    # Generate training data
    # Generate training data (same as before)
    t_train_LF = np.linspace(0, 1, N_LF).reshape(-1, 1)
    y_train_LF = y_LF(t_train_LF)
    t_train_HF = np.linspace(0, 1, N_HF).reshape(-1, 1)
    y_train_HF = y_HF(t_train_HF)

    # Convert to TensorFlow tensors
    t_train_HF_tf = tf.constant(t_train_HF, dtype=tf.float32)
    y_train_HF_tf = tf.constant(y_train_HF, dtype=tf.float32)

    # Collocation points for physics loss
    t_physics = tf.constant(np.linspace(0, 1, 500).reshape(-1, 1), dtype=tf.float32)

    # Build model for discrepancy
    model = keras.Sequential([
        keras.layers.Dense(50, activation='tanh', input_shape=(1,)),
        keras.layers.Dense(50, activation='tanh'),
        keras.layers.Dense(50, activation='tanh'),
        keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Use same physics parameters as Bayesian NN
    lambda_physics = physics_weight  # Now 0.5 instead of 0.1

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            # Data loss at HF points
            delta_pred = model(t_train_HF_tf)
            y_pred_HF = y_LF(t_train_HF_tf) + delta_pred
            loss_data = tf.reduce_mean(tf.square(y_pred_HF - y_train_HF_tf))

            # Physics loss with same formulation as Bayesian NN
            def total_solution(t):
                return y_LF(t) + model(t)

            physics_res = physics_residual_tf(t_physics, total_solution)
            
            # Match Bayesian NN: Gaussian likelihood with std=10.0
            # This is equivalent to weighted MSE: MSE/var where var = physics_std^2
            physics_variance = physics_std ** 2
            loss_physics = tf.reduce_mean(tf.square(physics_res)) / physics_variance

            # Total loss with consistent weighting
            total_loss = loss_data + lambda_physics * loss_physics

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total_loss, loss_data, loss_physics

    # Training
    print(f"Training Consistent Standard NN with {N_HF} HF samples...")
    for epoch in range(2000):
        total_loss, loss_d, loss_p = train_step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss:.4f}, Data Loss = {loss_d:.4f}, Physics Loss = {loss_p:.4f}")

    # Evaluation (same as before)
    t_test = np.linspace(0, 1, 500).reshape(-1, 1)
    y_true = y_HF(t_test)
    t_test_tf = tf.constant(t_test, dtype=tf.float32)
    delta_pred = model(t_test_tf).numpy()
    y_pred = y_LF(t_test) + delta_pred

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    complexity = 1000 * N_HF + 1 * N_LF

    # Calculate TRUE discrepancy for comparison
    lr = LinearRegression()
    lr.fit(y_LF(t_train_HF).reshape(-1, 1), y_train_HF)
    rho = float(lr.coef_[0])
    true_delta = y_HF(t_test) - rho * y_LF(t_test)

    if plot_results:
        # Create Nature-style Figure
        fig = plt.figure(figsize=(7, 5.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

        # Panel a: True HF vs. Prediction
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(t_test, y_true, color='#1f77b4', linewidth=2.5, label='True HF', zorder=3)
        ax1.plot(t_test, y_pred, color='#9467bd', linewidth=1.8, linestyle='--', label='Standard NN Prediction', zorder=4)
        ax1.scatter(t_train_HF, y_train_HF, s=60, color='#d62728', marker='D',
                    label=f'HF Data ($N_{{HF}}$={N_HF})', zorder=5, edgecolors='white', linewidth=0.8)
        ax1.scatter(t_train_LF, y_train_LF, s=8, color='#2ca02c', marker='o', alpha=0.6,
                    label=f'LF Data ($N_{{LF}}$={N_LF})', zorder=1)
        ax1.set_ylabel('Quantity of Interest, $y$', fontsize=11)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-3, 3)
        ax1.legend(loc='best', mode="expand", ncol=2, fontsize=9)
        ax1.text(0.02, 0.99, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')
        ax1.set_title(f'Case 2: Moderate Nonlinearities | Standard NN | $N_{{HF}}$={N_HF} | RMSE: {rmse:.4f} | Complexity: {complexity}',
                      fontsize=11, pad=10)

        # Panel b: Discrepancy Modeling
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(t_test, true_delta, color='#1f77b4', linewidth=2.5,
                 label='True $\\delta_{true}(t)$', alpha=0.8)
        ax2.plot(t_test, delta_pred, color='#9467bd', linewidth=2, linestyle='--',
                 label='Learned $\\delta_{pred}(t)$')
        # ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Input, $t$', fontsize=11)
        ax2.set_ylabel('Discrepancy, $\\delta$', fontsize=11)
        ax2.legend(bbox_to_anchor=(0.0, 0.95), loc='upper left', fontsize=9)
        ax2.set_xlim(0, 1)
        ax2.text(0.02, 0.99, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
        ax2.set_title('Standard NN Discrepancy Learning: True vs Predicted', fontsize=11, pad=10)

        plt.savefig(f'Standard_NN_NHF{N_HF}.png', dpi=600, bbox_inches='tight', facecolor='white')
        plt.show()

    return {
        'rmse': rmse,
        'complexity': complexity,
        'N_HF': N_HF,
        'N_LF': N_LF,
        'method': 'Standard NN'
    }

# ======================
# CASE 4: ANSATZ NN (WITH PHYSICS)
# ======================
def ansatz_NN(N_HF, N_LF, physics_weight=0.5, physics_std=10.0, plot_results=True):
    """
    Physics-Informed Neural Network with Analytical Ansatz and Physics Loss
    Case: Linear Scaling
    """
    # Generate training data
    t_train_LF = np.linspace(0, 1, N_LF).reshape(-1, 1)
    y_train_LF = y_LF(t_train_LF)
    t_train_HF = np.linspace(0, 1, N_HF).reshape(-1, 1)
    y_train_HF = y_HF(t_train_HF)

    # Convert to TensorFlow tensors
    t_train_HF_tf = tf.constant(t_train_HF, dtype=tf.float32)
    y_train_HF_tf = tf.constant(y_train_HF, dtype=tf.float32)

    # Collocation points for physics loss
    t_physics = tf.constant(np.linspace(0, 1, 500).reshape(-1, 1), dtype=tf.float32)

    # Define trainable parameters for analytical ansatz
    A = tf.Variable(1.0, dtype=tf.float32, name='amplitude')
    omega = tf.Variable(20.0, dtype=tf.float32, name='frequency')
    phi = tf.Variable(0.0, dtype=tf.float32, name='phase')
    B = tf.Variable(1.0, dtype=tf.float32, name='slope')
    C = tf.Variable(0.0, dtype=tf.float32, name='intercept')

    # Build neural network for residual correction
    nn_input = keras.layers.Input(shape=(1,))
    x = keras.layers.Dense(50, activation='tanh')(nn_input)
    x = keras.layers.Dense(50, activation='tanh')(x)
    x = keras.layers.Dense(50, activation='tanh')(x)
    nn_output = keras.layers.Dense(1)(x)
    nn_model = keras.Model(nn_input, nn_output)

    # Define the complete ansatz model
    def ansatz_model(t):
        base_solution = A * tf.sin(omega * t + phi) + B * t + C
        correction = nn_model(t)
        return base_solution + correction

    # Training with consistent physics
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    lambda_physics = physics_weight  # 0.5 to match Bayesian

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            # Data loss at HF points
            y_pred_HF = ansatz_model(t_train_HF_tf)
            loss_data = tf.reduce_mean(tf.square(y_pred_HF - y_train_HF_tf))

            # Physics loss with same formulation as Bayesian NN
            physics_res = physics_residual_tf(t_physics, ansatz_model)
            
            # Match Bayesian NN: Gaussian likelihood with std=10.0
            physics_variance = physics_std ** 2
            loss_physics = tf.reduce_mean(tf.square(physics_res)) / physics_variance

            # Total loss with consistent weighting
            total_loss = loss_data + lambda_physics * loss_physics

        trainable_vars = [A, omega, phi, B, C] + nn_model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))
        return total_loss, loss_data, loss_physics

    print(f"Training Consistent Ansatz NN with {N_HF} HF samples...")
    for epoch in range(2000):
        total_loss, loss_d, loss_p = train_step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss:.4f}, Data Loss = {loss_d:.4f}, Physics Loss = {loss_p:.4f}")

    # Evaluation
    t_test = np.linspace(0, 1, 500).reshape(-1, 1)
    y_true = y_HF(t_test)
    t_test_tf = tf.constant(t_test, dtype=tf.float32)
    y_pred = ansatz_model(t_test_tf).numpy()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    complexity = 1000 * N_HF + 1 * N_LF

    # Calculate discrepancy for comparison
    lr = LinearRegression()
    lr.fit(y_LF(t_train_HF).reshape(-1, 1), y_train_HF)
    rho = float(lr.coef_[0])
    true_delta = y_HF(t_test) - rho * y_LF(t_test)
    effective_delta = y_pred.flatten() - rho * y_LF(t_test).flatten()

    if plot_results:
        # Create Nature-style Figure for Ansatz NN
        fig = plt.figure(figsize=(7, 5.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

        # Panel a: True HF vs. Prediction
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(t_test, y_true, color='#1f77b4', linewidth=2.5, label='True HF', zorder=3)
        ax1.plot(t_test, y_pred, color='m', linewidth=1.8, linestyle='--', label='Ansatz NN Prediction', zorder=4)
        ax1.scatter(t_train_HF, y_train_HF, s=60, color='#d62728', marker='D',
                    label=f'HF Data ($N_{{HF}}$={N_HF})', zorder=5, edgecolors='white', linewidth=0.8)
        ax1.scatter(t_train_LF, y_train_LF, s=8, color='#2ca02c', marker='o', alpha=0.6,
                    label=f'LF Data ($N_{{LF}}$={N_LF})', zorder=1)
        ax1.set_ylabel('Quantity of Interest, $y$', fontsize=11)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-3, 3)
        ax1.legend(loc='best', mode="expand", ncol=2, fontsize=9)
        ax1.text(0.02, 0.99, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')
        ax1.set_title(f'Case 2: Moderate Nonlinearities | Ansatz NN | $N_{{HF}}$={N_HF} | RMSE: {rmse:.4f} | Complexity: {complexity}',
                      fontsize=11, pad=10)

        # Panel b: Discrepancy Modeling
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(t_test, true_delta, color='#1f77b4', linewidth=2.5,
                 label='True $\\delta_{true}(t)$', alpha=0.8)
        ax2.plot(t_test, effective_delta, color='m', linewidth=2, linestyle='--',
                 label='Learned $\\delta_{pred}(t)$')
        # ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Input, $t$', fontsize=11)
        ax2.set_ylabel('Discrepancy, $\\delta$', fontsize=11)
        ax2.legend(bbox_to_anchor=(0.0, 0.95), loc='upper left', fontsize=9)
        ax2.set_xlim(0, 1)
        ax2.text(0.02, 0.99, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
        ax2.set_title('Ansatz NN Discrepancy Learning: True vs Predicted', fontsize=11, pad=10)

        plt.savefig(f'Ansatz_NN_NHF{N_HF}.png', dpi=600, bbox_inches='tight', facecolor='white')
        plt.show()

    return {
        'rmse': rmse,
        'complexity': complexity,
        'N_HF': N_HF,
        'N_LF': N_LF,
        'method': 'Ansatz NN',
        'learned_params': {
            'A': A.numpy(),
            'omega': omega.numpy(),
            'phi': phi.numpy(),
            'B': B.numpy(),
            'C': C.numpy()
        }
    }

# ======================
# BAYESIAN ANSATZ NN (B-NN)
# ======================
def bayesian_ansatz_NN(N_HF, N_LF, noise_std=0.1, n_chains=1, num_results=80, num_burnin_steps=20, plot_results=True):
    """
    Bayesian Physics-Informed Neural Network with Analytical Ansatz
    Uses HMC for posterior estimation and uncertainty quantification
    """
    # Generate training data with noise
    np.random.seed(42)
    t_train_LF = np.linspace(0, 1, N_LF).reshape(-1, 1)
    y_train_LF = y_LF(t_train_LF) + np.random.normal(0, noise_std, t_train_LF.shape)
    t_train_HF = np.linspace(0, 1, N_HF).reshape(-1, 1)
    y_train_HF = y_HF(t_train_HF) + np.random.normal(0, noise_std, t_train_HF.shape)

    # Collocation points for physics loss
    t_physics = np.linspace(0, 1, 100).reshape(-1, 1)
    
    # Convert to TensorFlow tensors
    t_train_HF_tf = tf.constant(t_train_HF, dtype=tf.float32)
    y_train_HF_tf = tf.constant(y_train_HF, dtype=tf.float32)
    t_physics_tf = tf.constant(t_physics, dtype=tf.float32)

    # Bayesian model definition - SIMPLIFIED VERSION
    def make_bayesian_ansatz_model():
        """More reasonable priors"""
        # Wider priors for analytical parameters
        A = tfd.Normal(loc=1.0, scale=2.0, name='A')
        omega = tfd.Normal(loc=20.0, scale=5.0, name='omega')  
        phi = tfd.Normal(loc=0.0, scale=1.0, name='phi')  
        B = tfd.Normal(loc=1.0, scale=1.0, name='B')  
        C = tfd.Normal(loc=0.0, scale=1.0, name='C')  
        
        # Neural network parameters with reasonable scales
        nn_weight = tfd.Normal(loc=0.0, scale=1.0, name='nn_weight')  
        nn_bias = tfd.Normal(loc=0.0, scale=1.0, name='nn_bias')  
        
        # noise level
        noise_std_prior = tfd.HalfNormal(scale=0.5, name='noise_std')
        
        return tfd.JointDistributionNamed({
            'A': A, 'omega': omega, 'phi': phi, 'B': B, 'C': C,
            'nn_weight': nn_weight, 'nn_bias': nn_bias, 'noise_std': noise_std_prior
        })

    # Define the Bayesian ansatz model - SIMPLIFIED
    def bayesian_ansatz(t, params):
        """Bayesian ansatz model with simplified neural network correction"""
        # Analytical part
        analytical_part = (
            params['A'] * tf.sin(params['omega'] * t + params['phi']) + 
            params['B'] * t + params['C']
        )
        
        # Simplified neural network part - single layer
        nn_correction = params['nn_weight'] * tf.sin(40 * t) + params['nn_bias'] * tf.cos(40 * t)
        
        return analytical_part + 0.1 * nn_correction

    # Physics residual for Bayesian model
    def physics_residual_bayesian(t, params):
        """Compute physics residual for Bayesian model"""
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t)
            with tf.GradientTape() as tape1:
                tape1.watch(t)
                y_val = bayesian_ansatz(t, params)
            dy_dt = tape1.gradient(y_val, t)
        d2y_dt2 = tape2.gradient(dy_dt, t)
        del tape2
        
        return d2y_dt2 + 400.0 * y_val - 400.0 * t

    # Define the target log probability for HMC
    def target_log_prob_fn(A, omega, phi, B, C, nn_weight, nn_bias, noise_std):
        """Improved target log probability"""
        params = {
            'A': A, 'omega': omega, 'phi': phi, 'B': B, 'C': C, 
            'nn_weight': nn_weight, 'nn_bias': nn_bias
        }
        
        # Prior contribution
        model = make_bayesian_ansatz_model()
        log_prior = model.log_prob({
            'A': A, 'omega': omega, 'phi': phi, 'B': B, 'C': C, 
            'nn_weight': nn_weight, 'nn_bias': nn_bias, 'noise_std': noise_std
        })
        
        # Data likelihood - use learned noise_std instead of fixed value
        y_pred = bayesian_ansatz(t_train_HF_tf, params)
        log_likelihood_data = tf.reduce_sum(
            tfd.Normal(loc=y_pred, scale=noise_std).log_prob(y_train_HF_tf)
        )
        
        # Physics likelihood with more reasonable uncertainty
        physics_res = physics_residual_bayesian(t_physics_tf, params)
        physics_std = 8.0  # Increased - allows more physics violation
        log_likelihood_physics = tf.reduce_sum(
            tfd.Normal(loc=0.0, scale=physics_std).log_prob(physics_res)
        )
        
        # Better balanced weights
        physics_weight = 0.5  # Increased from 0.1
        return log_prior + log_likelihood_data + physics_weight * log_likelihood_physics

    # HMC sampling - FIXED VERSION
    print(f"Running Bayesian Ansatz NN with {N_HF} HF samples (HMC sampling)...")
    
    # Initial state - as a list for HMC
    initial_state = [
        tf.constant(1.0, dtype=tf.float32),   # A
        tf.constant(20.0, dtype=tf.float32),  # omega
        tf.constant(0.0, dtype=tf.float32),   # phi
        tf.constant(1.0, dtype=tf.float32),   # B
        tf.constant(0.0, dtype=tf.float32),   # C
        tf.constant(0.1, dtype=tf.float32),   # nn_weight
        tf.constant(0.1, dtype=tf.float32),   # nn_bias
        tf.constant(noise_std, dtype=tf.float32)  # Start with actual noise level
    ]

    # Define unconstraining bijectors - all parameters as scalars
    unconstraining_bijectors = [
        tfb.Identity(),   # A
        tfb.Identity(),   # omega  
        tfb.Identity(),   # phi
        tfb.Identity(),   # B
        tfb.Identity(),   # C
        tfb.Identity(),   # nn_weight
        tfb.Identity(),   # nn_bias
        tfb.Exp()        # noise_std (positive)
    ]

    # HMC kernel
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.005,  # Smaller step size for better mixing
        num_leapfrog_steps=5  # More leapfrog steps
    )
    
    # Transformed transition kernel
    transformed_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=hmc_kernel,
        bijector=unconstraining_bijectors
    )

    # Run HMC sampling - WITHOUT @tf.function to avoid AutoGraph issues
    def run_chain(initial_state):
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=transformed_kernel,
            trace_fn=lambda _, pkr: pkr)
        return samples, kernel_results

    # Run multiple chains
    print("Running HMC sampling...")
    all_samples = []
    for chain in range(n_chains):
        print(f"Chain {chain + 1}/{n_chains}")
        samples, kernel_results = run_chain(initial_state)
        all_samples.append(samples)
    
    # Calculate acceptance rate
    acceptance_rate = tf.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, tf.float32))
    print(f"Acceptance rate: {acceptance_rate.numpy():.3f}")
    
    # Combine chains
    combined_samples = []
    for i in range(len(all_samples[0])):
        chain_samples = [samples[i] for samples in all_samples]
        combined_samples.append(tf.concat(chain_samples, axis=0))

    # Posterior predictions
    print("Computing posterior predictions...")
    t_test = np.linspace(0, 1, 200).reshape(-1, 1)
    t_test_tf = tf.constant(t_test, dtype=tf.float32)
    
    # Generate posterior predictions
    n_posterior_samples = 200
    posterior_predictions = []
    delta_predictions = []  # NEW: Store discrepancy predictions
    analytical_params = []
    
    # Calculate rho for discrepancy (same as other methods)
    lr = LinearRegression()
    lr.fit(y_LF(t_train_HF).reshape(-1, 1), y_train_HF)
    rho = float(lr.coef_[0])
    
    for i in range(n_posterior_samples):
        sample_idx = np.random.randint(0, combined_samples[0].shape[0])
        params = {
            'A': combined_samples[0][sample_idx],
            'omega': combined_samples[1][sample_idx], 
            'phi': combined_samples[2][sample_idx],
            'B': combined_samples[3][sample_idx],
            'C': combined_samples[4][sample_idx],
            'nn_weight': combined_samples[5][sample_idx],
            'nn_bias': combined_samples[6][sample_idx]
        }
        
        y_pred = bayesian_ansatz(t_test_tf, params)
        posterior_predictions.append(y_pred.numpy())
        
        # NEW: Calculate discrepancy for this sample
        delta_pred = y_pred.numpy() - rho * y_LF(t_test)
        delta_predictions.append(delta_pred)
        
        analytical_params.append([params['A'].numpy(), params['omega'].numpy(), 
                                params['phi'].numpy(), params['B'].numpy(), params['C'].numpy()])
    
    posterior_predictions = np.array(posterior_predictions)
    delta_predictions = np.array(delta_predictions)  # NEW: Array of discrepancy predictions
    analytical_params = np.array(analytical_params)
    
    # Compute statistics for predictions
    y_pred_mean = np.mean(posterior_predictions, axis=0)
    y_pred_std = np.std(posterior_predictions, axis=0)
    y_pred_upper = y_pred_mean + 2 * y_pred_std
    y_pred_lower = y_pred_mean - 2 * y_pred_std
    
    # NEW: Compute statistics for discrepancy
    effective_delta_mean = np.mean(delta_predictions, axis=0)
    effective_delta_std = np.std(delta_predictions, axis=0)
    effective_delta_high = effective_delta_mean + 2 * effective_delta_std
    effective_delta_low = effective_delta_mean - 2 * effective_delta_std
    
    # Calculate RMSE
    y_true = y_HF(t_test)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_mean)))
    complexity = 1000 * N_HF + 1 * N_LF

    # NEW: Calculate true discrepancy for comparison
    true_delta = y_HF(t_test) - rho * y_LF(t_test)

    if plot_results:
        # Create comprehensive visualization
        fig = plt.figure(figsize=(7, 5.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
        
        # Panel a: Posterior predictions with uncertainty
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(t_test, y_true, color='#1f77b4', linewidth=2.5, label='True HF', zorder=3)
        ax1.plot(t_test, y_pred_mean, color='#e377c2', linewidth=1.8, linestyle='--', 
                label='Bayesian Ansatz NN', zorder=4)
        ax1.fill_between(t_test.flatten(), y_pred_lower.flatten(), y_pred_upper.flatten(),
                color='#e377c2', alpha=0.3, label='95% CI', zorder=2)
        ax1.scatter(t_train_HF, y_train_HF, s=60, color='#d62728', marker='D', 
                   label=f'HF Data ($N_{{HF}}$={N_HF})', zorder=5, edgecolors='white', linewidth=0.8)
        ax1.scatter(t_train_LF, y_train_LF, s=8, color='#2ca02c', marker='o', alpha=0.6, 
                   label=f'LF Data ($N_{{LF}}$={N_LF})', zorder=1)
        ax1.set_ylabel('Quantity of Interest, $y$', fontsize=11)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-3, 3)
        ax1.legend(loc='best', mode="expand", ncol=2, fontsize=9)
        ax1.text(0.02, 0.99, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')
        ax1.set_title(f'Case 2: Moderate Nonlinearities | Bayesian Ansatz NN | $N_{{HF}}$={N_HF} | RMSE: {rmse:.4f} | Complexity: {complexity}', 
                     fontsize=11, pad=10)
        
        # Panel b: Discrepancy comparison with uncertainty
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(t_test, true_delta, color='#1f77b4', linewidth=2.5, 
                label='True $\delta_{true}(t)$', alpha=0.8)
        ax2.plot(t_test, effective_delta_mean, color='#e377c2', linewidth=2, linestyle='--', 
                label='Learned $\delta_{pred}(t)$')
        ax2.fill_between(t_test.squeeze(-1), effective_delta_low.squeeze(-1), effective_delta_high.squeeze(-1),
                        color='#e377c2', alpha=0.3, label='95% CI')
        # ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Input, $t$', fontsize=11)
        ax2.set_ylabel('Discrepancy, $\delta$', fontsize=11)
        ax2.legend(bbox_to_anchor=(0.0, 0.95), loc='upper left', fontsize=9)
        ax2.set_xlim(0, 1)
        ax2.text(0.02, 0.99, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
        ax2.set_title('Bayesian Ansatz NN Discrepancy Learning: True vs Predicted', fontsize=11, pad=10)

        plt.savefig(f'Bayesian_Ansatz_NN_NHF{N_HF}.png', dpi=600, bbox_inches='tight', facecolor='white')
        plt.show()

    return {
        'rmse': rmse,
        'complexity': complexity,
        'N_HF': N_HF,
        'N_LF': N_LF,
        'noise_std': noise_std,
        'method': 'Bayesian Ansatz NN (Fixed)',
        'predictive_mean': y_pred_mean,
        'predictive_std': y_pred_std,
        'parameter_posteriors': analytical_params,
        'acceptance_rate': acceptance_rate.numpy()
    }

# ======================
# UPDATED MAIN EXECUTION WITH ALL METHODS
# ======================
if __name__ == "__main__":
    print("Running Comprehensive Multi-fidelity NN Analysis...")
    print("=" * 70)

    # Sample sizes to test
    n_hf_samples = [5, 10, 15]
    n_lf_samples = 1000
    
    # Consistent physics parameters
    PHYSICS_WEIGHT = 0.5
    PHYSICS_STD = 8.0

    # Store results for all methods
    standard_NN_results = []
    ansatz_NN_results = []
    bayesian_NN_results = []  # HMC version

    # Run Standard NN for each sample size
    print("\nRunning Standard NN...")
    for n_hf in n_hf_samples:
        print(f"  N_HF = {n_hf}...")
        results = standard_NN(N_HF=n_hf, N_LF=n_lf_samples, 
                                          physics_weight=PHYSICS_WEIGHT, 
                                          physics_std=PHYSICS_STD, 
                                          plot_results=True)
        standard_NN_results.append(results)
        print(f"    RMSE: {results['rmse']:.4f}, Complexity: {results['complexity']}")

    # Run Ansatz NN for each sample size
    print("\nRunning Ansatz NN...")
    for n_hf in n_hf_samples:
        print(f"  N_HF = {n_hf}...")
        results = ansatz_NN(N_HF=n_hf, N_LF=n_lf_samples, 
                                        physics_weight=PHYSICS_WEIGHT, 
                                        physics_std=PHYSICS_STD, 
                                        plot_results=True)
        ansatz_NN_results.append(results)
        print(f"    RMSE: {results['rmse']:.4f}, Complexity: {results['complexity']}")
        print(f"    Learned params: A={results['learned_params']['A']:.3f}, ω={results['learned_params']['omega']:.3f}")

    # Run Bayesian Ansatz NN (HMC) for each sample size
    print("\nRunning Bayesian Ansatz NN (HMC)...")
    for n_hf in n_hf_samples:
        print(f"  N_HF = {n_hf}...")
        results = bayesian_ansatz_NN(N_HF=n_hf, N_LF=n_lf_samples, noise_std=0.0, plot_results=True)
        bayesian_NN_results.append(results)
        print(f"    RMSE: {results['rmse']:.4f}, Complexity: {results['complexity']}")

    # ======================
    # COMPREHENSIVE COMPARISON PLOT (ALL FOUR METHODS)
    # ======================
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Extract results
    standard_rmses = [r['rmse'] for r in standard_NN_results]
    ansatz_rmses = [r['rmse'] for r in ansatz_NN_results]
    bayesian_rmses = [r['rmse'] for r in bayesian_NN_results]

    # Plot all four methods together
    ax.plot(n_hf_samples, standard_rmses, 'o-', color='#9467bd', linewidth=2.5, 
             markersize=8, label='Standard NN', markerfacecolor='white', markeredgewidth=2)
    ax.plot(n_hf_samples, ansatz_rmses, 's-', color='m', linewidth=2.5, 
             markersize=8, label='Ansatz NN', markerfacecolor='white', markeredgewidth=2)
    ax.plot(n_hf_samples, bayesian_rmses, 'D-', color='#e377c2', linewidth=2.5, 
             markersize=8, label='Bayesian Ansatz NN', markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Number of HF Samples', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Case 2: Moderate Nonlinearities | NN Methods Comparison: RMSE vs Data Quantity', 
                fontsize=13, pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('ALL_METHODS_RMSE_Comparison.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()

    # ======================
    # COMPREHENSIVE RESULTS SUMMARY
    # ======================
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON SUMMARY: ALL FOUR METHODS")
    print("=" * 80)

    print(f"\n{'Method':<30} {'N_HF':<6} {'RMSE':<10} {'Complexity':<12}")
    print("-" * 80)

    for i, n_hf in enumerate(n_hf_samples):
        std = standard_NN_results[i]
        ansatz = ansatz_NN_results[i]
        bayes = bayesian_NN_results[i]
        
        print(f"{'Standard NN':<30} {n_hf:<6} {std['rmse']:<10.4f} {std['complexity']:<12}")
        print(f"{'Ansatz NN':<30} {n_hf:<6} {ansatz['rmse']:<10.4f} {ansatz['complexity']:<12}")
        print(f"{'Bayesian NN (HMC)':<30} {n_hf:<6} {bayes['rmse']:<10.4f} {bayes['complexity']:<12}")
        print("-" * 80)

    # Performance analysis
    print("\nPERFORMANCE ANALYSIS:")
    print("-" * 50)