import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# ======================
# PLOTTING
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

# Define the HF and LF functions
def y_HF(t):
    return np.sin(20 * t) + t

def y_LF(t):
    return 0.8 * np.sin(20 * t) + 0.5 * t + 0.5

# ======================
# CASE 1: STANDARD GP (AUTOREGRESSIVE CO-KRIGING)
# ======================
def standard_gp_co_kriging(N_HF, N_LF, plot_results=True):
    """
    Standard GP with autoregressive scheme: y_HF(t) = ρ * y_LF(t) + δ(t)
    Case: Linear Scaling
    """
    # Generate training data
    t_train_LF = np.linspace(0, 1, N_LF).reshape(-1, 1)
    y_train_LF = y_LF(t_train_LF)
    t_train_HF = np.linspace(0, 1, N_HF).reshape(-1, 1)
    y_train_HF = y_HF(t_train_HF)

    # Step 1: Train GP on LF data
    kernel_LF = ConstantKernel(1.0) * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.01)
    gp_LF = GaussianProcessRegressor(kernel=kernel_LF, n_restarts_optimizer=5, random_state=42)
    gp_LF.fit(t_train_LF, y_train_LF)

    # Step 2: Estimate correlation and discrepancy
    lr = LinearRegression()
    lr.fit(y_LF(t_train_HF).reshape(-1, 1), y_train_HF)
    rho_estimate = lr.coef_[0]
    rho = float(rho_estimate)
    delta_train = y_train_HF - rho * y_LF(t_train_HF)

    # Step 3: Train GP on the discrepancy
    kernel_delta = ConstantKernel(1.0) * RBF(length_scale=0.05) + WhiteKernel(noise_level=0.001)
    gp_delta = GaussianProcessRegressor(kernel=kernel_delta, n_restarts_optimizer=5, random_state=42)
    gp_delta.fit(t_train_HF, delta_train)

    # Step 4: Make predictions
    t_test = np.linspace(0, 1, 500).reshape(-1, 1)
    y_LF_pred, y_LF_std = gp_LF.predict(t_test, return_std=True)
    delta_pred, delta_std = gp_delta.predict(t_test, return_std=True)
    y_HF_pred = rho * y_LF_pred + delta_pred
    y_HF_std = np.sqrt((rho * y_LF_std)**2 + delta_std**2)

    y_true = y_HF(t_test)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_HF_pred)))
    complexity = 1000 * N_HF + 1 * N_LF
    
    # Calculate TRUE discrepancy for comparison
    true_delta = y_HF(t_test) - rho * y_LF(t_test)

    if plot_results:
        # Create Nature-style Figure
        fig = plt.figure(figsize=(7, 5.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

        # Panel a: True HF vs. Prediction
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(t_test, y_true, color='#1f77b4', linewidth=2.5, label='True HF', zorder=3)
        ax1.plot(t_test, y_HF_pred, color='#d62728', linewidth=1.8, linestyle='--', label='Standard GP Prediction', zorder=4)
        ax1.fill_between(t_test.flatten(),
                         y_HF_pred.flatten() - 2 * y_HF_std,
                         y_HF_pred.flatten() + 2 * y_HF_std,
                         color='#d62728', alpha=0.25, label='95% CI', zorder=2)
        ax1.scatter(t_train_HF, y_train_HF, s=60, color='#d62728', marker='D', 
                    label=f'HF Data ($N_{{HF}}$={N_HF})', zorder=5, edgecolors='white', linewidth=0.8)
        ax1.scatter(t_train_LF, y_train_LF, s=8, color='#2ca02c', marker='o', alpha=0.6, 
                    label=f'LF Data ($N_{{LF}}$={N_LF})', zorder=1)
        ax1.set_ylabel('Quantity of Interest, $y$', fontsize=11)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-3, 3)
        ax1.legend(loc='best', mode="expand", ncol=2, fontsize=9)
        ax1.text(0.02, 0.99, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')
        ax1.set_title(f'Case 1: Linear Scaling | Standard GP | $N_{{HF}}$={N_HF} | RMSE: {rmse:.4f} | Complexity: {complexity}', 
                      fontsize=11, pad=10)

        # Panel b: Discrepancy Modeling
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(t_test, true_delta, color='#1f77b4', linewidth=2.5, 
                 label='True $\delta_{true}(t)$', alpha=0.8)
        ax2.plot(t_test, delta_pred, color='#d62728', linewidth=2, linestyle='--',
                 label='Learned $\delta_{pred}(t)$')
        ax2.fill_between(t_test.flatten(),
                         delta_pred.flatten() - 2 * delta_std,
                         delta_pred.flatten() + 2 * delta_std,
                         color='#d62728', alpha=0.3, label='95% CI')
        # ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Input, $t$', fontsize=11)
        ax2.set_ylabel('Discrepancy, $\delta$', fontsize=11)
        ax2.legend(bbox_to_anchor=(0.0, 0.95), loc='upper left', fontsize=9)
        ax2.set_xlim(0, 1)
        ax2.text(0.02, 0.99, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
        ax2.set_title('Standard GP Discrepancy Learning: True vs Predicted', fontsize=11, pad=10)

        plt.savefig(f'Standard_GP_NHF{N_HF}.png', dpi=600, bbox_inches='tight', facecolor='white')
        plt.show()

    return {
        'rmse': rmse,
        'complexity': complexity,
        'rho': rho,
        'N_HF': N_HF,
        'N_LF': N_LF,
        'method': 'Standard GP'
    }

# ======================
# CASE 2: JOINT GP (ADVANCED CO-KRIGING)
# ======================
def joint_gp_co_kriging(N_HF, N_LF, plot_results=True):
    """
    Joint GP with enhanced kernel structure for nonlinear relationships
    Case: Linear Scaling
    """
    # Generate training data
    t_train_LF = np.linspace(0, 1, N_LF).reshape(-1, 1)
    y_train_LF = y_LF(t_train_LF)
    t_train_HF = np.linspace(0, 1, N_HF).reshape(-1, 1)
    y_train_HF = y_HF(t_train_HF)

    # Create augmented dataset for multi-fidelity GP
    X_train = np.vstack([
        np.hstack([t_train_LF, np.zeros((N_LF, 1))]),  # LF data: fidelity = 0
        np.hstack([t_train_HF, np.ones((N_HF, 1))])    # HF data: fidelity = 1
    ])
    y_train = np.concatenate([y_train_LF, y_train_HF])

    # Enhanced kernel structure for nonlinear relationships
    kernel_t = RBF(length_scale=0.1) + RBF(length_scale=0.02)  # Multiple length scales
    kernel_fidelity = ConstantKernel(1.0) * RBF(length_scale=1.0)  # Nonlinear fidelity relationship
    
    from sklearn.gaussian_process.kernels import Product
    kernel = Product(kernel_t, kernel_fidelity) + WhiteKernel(noise_level=0.01)

    # Train joint multi-fidelity GP
    gp_mf = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, random_state=42)
    gp_mf.fit(X_train, y_train)

    # Make predictions for HF level
    t_test = np.linspace(0, 1, 500).reshape(-1, 1)
    X_test = np.hstack([t_test, np.ones((500, 1))])  # Predict at HF fidelity
    y_pred, y_std = gp_mf.predict(X_test, return_std=True)
    
    y_true = y_HF(t_test)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred.flatten())))
    complexity = 1000 * N_HF + 1 * N_LF

    # Calculate discrepancy for Joint GP (approximate since it's joint modeling)
    lr = LinearRegression()
    lr.fit(y_LF(t_train_HF).reshape(-1, 1), y_train_HF)
    rho_approx = float(lr.coef_[0])
    effective_delta = y_pred.flatten() - rho_approx * y_LF(t_test).flatten()
    true_delta_approx = y_HF(t_test).flatten() - rho_approx * y_LF(t_test).flatten()

    if plot_results:
        # Create Nature-style Figure for Joint GP
        fig = plt.figure(figsize=(7, 5.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

        # Panel a: True HF vs. Prediction
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(t_test, y_true, color='#1f77b4', linewidth=2.5, label='True HF', zorder=3)
        ax1.plot(t_test, y_pred, color='#ff7f0e', linewidth=1.8, linestyle='--', label='Joint GP Prediction', zorder=4)
        ax1.fill_between(t_test.flatten(),
                         y_pred.flatten() - 2 * y_std,
                         y_pred.flatten() + 2 * y_std,
                         color='#ff7f0e', alpha=0.25, label='95% CI', zorder=2)
        ax1.scatter(t_train_HF, y_train_HF, s=60, color='#d62728', marker='D', 
                    label=f'HF Data ($N_{{HF}}$={N_HF})', zorder=5, edgecolors='white', linewidth=0.8)
        ax1.scatter(t_train_LF, y_train_LF, s=8, color='#2ca02c', marker='o', alpha=0.6, 
                    label=f'LF Data ($N_{{LF}}$={N_LF})', zorder=1)
        ax1.set_ylabel('Quantity of Interest, $y$', fontsize=11)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-3, 3)
        ax1.legend(loc='best', mode="expand", ncol=2, fontsize=9)
        ax1.text(0.02, 0.99, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')
        ax1.set_title(f'Case 1: Linear Scaling | Joint GP | $N_{{HF}}$={N_HF} | RMSE: {rmse:.4f} | Complexity: {complexity}', 
                      fontsize=11, pad=10)

        # Panel b: DISCREPANCY MODELING
        ax2 = fig.add_subplot(gs[1])
        
        # Plot TRUE discrepancy (approximate for Joint GP)
        ax2.plot(t_test, true_delta_approx, color='#1f77b4', linewidth=2.5, 
                 label='True $\delta_{true}(t)$', alpha=0.8)
        
        # Plot EFFECTIVE discrepancy learned by Joint GP
        ax2.plot(t_test, effective_delta, color='#ff7f0e', linewidth=2, 
                 label='Learned $\delta_{pred}(t)$', linestyle='--')
        
        # FIXED: Use y_std for uncertainty in discrepancy
        ax2.fill_between(t_test.flatten(),
                         effective_delta - 2 * y_std,
                         effective_delta + 2 * y_std,
                         color='#ff7f0e', alpha=0.3, label='95% CI')
        
        # Show training points
        delta_train_approx = y_train_HF.flatten() - rho_approx * y_LF(t_train_HF).flatten()
        # ax2.scatter(t_train_HF, delta_train_approx, s=40, color='#ff7f0e', marker='s', 
        #             zorder=5, edgecolors='white', linewidth=0.8, 
        #             label='Training Data')
        
        # ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Input, $t$', fontsize=11)
        ax2.set_ylabel('Discrepancy, $\delta$', fontsize=11)
        ax2.legend(bbox_to_anchor=(0.0, 0.95), loc='upper left', fontsize=9)
        ax2.set_xlim(0, 1)
        ax2.text(0.02, 0.99, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
        ax2.set_title('Joint GP Discrepancy Learning: True vs Predicted', fontsize=11, pad=10)

        plt.savefig(f'Joint_GP_NHF{N_HF}.png', dpi=600, bbox_inches='tight', facecolor='white')
        plt.show()

    return {
        'rmse': rmse,
        'complexity': complexity,
        'N_HF': N_HF,
        'N_LF': N_LF,
        'method': 'Joint GP'
    }

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    print("Running Multi-fidelity Gaussian Process Analysis...")
    print("=" * 70)
    
    # Sample sizes to test
    n_hf_samples = [5, 10, 15]
    n_lf_samples = 1000
    
    # Store results for comparison
    standard_gp_results = []
    joint_gp_results = []
    
    # Run Standard GP for each sample size
    print("\nRunning Standard GP (Linear Scaling)...")
    for n_hf in n_hf_samples:
        print(f"  N_HF = {n_hf}...")
        results = standard_gp_co_kriging(N_HF=n_hf, N_LF=n_lf_samples, plot_results=True)
        standard_gp_results.append(results)
        print(f"    RMSE: {results['rmse']:.4f}, Complexity: {results['complexity']}")
    
    # Run Joint GP for each sample size  
    print("\nRunning Joint GP (Nonlinear Scaling)...")
    for n_hf in n_hf_samples:
        print(f"  N_HF = {n_hf}...")
        results = joint_gp_co_kriging(N_HF=n_hf, N_LF=n_lf_samples, plot_results=True)
        joint_gp_results.append(results)
        print(f"    RMSE: {results['rmse']:.4f}, Complexity: {results['complexity']}")
    
    # ======================
    # COMPARISON PLOTS
    # ======================
#%%    
    # Plot: RMSE vs Number of Samples
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    # RMSE vs Samples
    standard_rmses = [r['rmse'] for r in standard_gp_results]
    joint_rmses = [r['rmse'] for r in joint_gp_results]
    
    ax.plot(n_hf_samples, standard_rmses, 'o-', color='#d62728', linewidth=2.5, 
             markersize=8, label='Standard GP', markerfacecolor='white', markeredgewidth=2)
    ax.plot(n_hf_samples, joint_rmses, 's-', color='#ff7f0e', linewidth=2.5, 
             markersize=8, label='Joint GP', markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Number of HF Samples', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Case 1: Linear Scaling | GP Methods Comparison: RMSE vs Data Quantity', fontsize=13, pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('GP_Comparison_Metrics.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # ======================
    # RESULTS SUMMARY
    # ======================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY: STANDARD GP vs JOINT GP")
    print("=" * 70)
    
    print(f"\n{'Method':<20} {'N_HF':<6} {'RMSE':<10} {'Complexity':<12} {'Scaling Type':<15}")
    print("-" * 70)
    
    for i, n_hf in enumerate(n_hf_samples):
        std = standard_gp_results[i]
        joint = joint_gp_results[i]
        
        print(f"{'Standard GP':<20} {n_hf:<6} {std['rmse']:<10.4f} {std['complexity']:<12} {'Linear':<15}")
        print(f"{'Joint GP':<20} {n_hf:<6} {joint['rmse']:<10.4f} {joint['complexity']:<12} {'Linear':<15}")
        print("-" * 70)
    
    # Calculate average improvements
    avg_rmse_improvement = np.mean([std['rmse'] - joint['rmse'] for std, joint in zip(standard_gp_results, joint_gp_results)])
    avg_rmse_improvement_pct = np.mean([(std['rmse'] - joint['rmse'])/std['rmse'] * 100 for std, joint in zip(standard_gp_results, joint_gp_results)])