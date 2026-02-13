from openpyxl.descriptors import Float
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


def gaussian_tail_exact(z: np.ndarray) -> np.ndarray:
    """
    Gaussian tail (CLT approximation): P(Z > z) for standard normal.
    """
    return 1 - stats.norm.cdf(z)


def gaussian_chernoff_bound(z: np.ndarray) -> np.ndarray:
    """
    Chernoff/Gaussian upper bound: \exp(-z^2/2)
    """
    return np.exp(-z**2 / 2)


def _hoeffding_bound(t: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """
    Wainwright (2.10)
    \Pr[\sum_{i=1}^n X_i > t] ≤ \exp(-\frac{t^2}{2\sum_{i=1}^n \sigma_i^2})
    sigma2 is the subgaussian parameter (not the variance)
    """
    return np.exp(-t**2 / (2*np.sum(sigma2)))


def hoeffding_bound_z(z, SE, N, sigma2) -> np.ndarray:
    """
    substitute t = z SE N
    """
    return _hoeffding_bound(z * SE * N, sigma2)


def _bernstein_bound(delta: np.ndarray, X2: np.ndarray, N: int, b: float = 1.0) -> np.ndarray:
    """
    Wainwright 2.22b
    \Pr[\sum_{i=1}^n X_i >= n \delta ] ≤ \exp(- n \frac{\delta^2}{2(1/n \sum_i \E[X_i^2] + b\delta /3)})
    """
    delta = np.asarray(delta)
    X2_mean = np.mean(X2)
    denom = 2 * (X2_mean + (b * delta) / 3.0)
    exponent = -N * (delta ** 2) / denom
    return np.exp(exponent)


def bernstein_bound(z, SE, N, X2, b=1.0) -> np.ndarray:
    """
    substitute delta = z SE
    """
    return _bernstein_bound(z * SE, X2, N, b)


def bennett_bound(z, SE, N, X2, b=1.0) -> np.ndarray:
    """
    Bennett's inequality (tighter than Bernstein for some regimes)
    """
    delta = z * SE
    delta = np.asarray(delta)
    X2_mean = np.mean(X2)
    v = X2_mean

    # Bennett: exp(-n * v / b^2 * h(b * delta / v))
    # where h(u) = (1+u) log(1+u) - u
    u = b * delta / v
    h = (1 + u) * np.log(1 + u + 1e-10) - u
    return np.exp(-N * v / (b**2) * h)

def binomial_exact(t: np.ndarray, N: int, p: float):
    """
    Pr[Binom(N, p) >= t]
    """
    return 1 - stats.binom.cdf(t, N, p)

def binomial_sample(t: np.ndarray, N: int, p: float, n_samples: int = 1_000_000):
    """
    Estimate Pr[Binom(N, p) >= t] by Monte Carlo sampling.
    """
    samples = np.random.binomial(N, p, size=n_samples)
    # shape: (n_samples, 1) >= (len(t),) broadcasts to (n_samples, len(t))
    return (samples[:, None] >= t[None, :]).mean(axis=0)


def _binomial_bound(delta: np.ndarray, N: int, p: float):
    """
    Wainwright Exercise 2.9
    If Z_i ~ Bernouli(p), iid, for p in (0, 1/2)
    P[\sum_i Z_i <= delta N] ≤ \exp(-N D(delta||p))
    """
    assert p > 0 and p < 0.5
    kl = delta * np.log(delta / p) + (1 - delta) * np.log((1 - delta) / (1 - p))
    return np.exp(-N * kl)

def binomial_bound(t: np.ndarray, N: int, p: float):
    """
    use symmetry to get P[\sum_i Z_i >= t] = P[\sum_i Z'_i <= N - t]
    
    where Z'_i ~ Bernouli(1-p)
    then apply the lowerbound of P[\sum_i Z'_i <= delta N] 
    """
    assert p > 0.5 and p < 1
    delta = 1 - t / N
    return _binomial_bound(delta, N, 1 - p)

def generate_iid_binomial_tailbounds(N: int, p: Float, z_max: float = 6.0, n_points: int = 10000):
    """
    Generate tail bounds data for plotting.

    Args:
        N: number of samples
        p: probability parameter for binomial
        z_max: maximum z-score to plot
        n_points: number of points for plotting

    Returns:
        tuple: (DataFrame with all bounds, metadata dict with sigma and N)
    """
    z = np.linspace(0.01, z_max, n_points)

    # Standard error
    sigma = np.sqrt(p*(1-p))
    SE = sigma / np.sqrt(N)

    # CLT estimates
    X2 = np.ones(N) * sigma**2  # E[X_i^2] for centered RVs
    # Create DataFrame
    df = pd.DataFrame({
        'z': z,
        'Gaussian (CLT)': gaussian_tail_exact(z),
        'Hoeffding': hoeffding_bound_z(z, SE, N, np.ones(N)*0.25),
        # center these RVs to get the strongest results from Bernstein and Bennett
        'Bernstein': bernstein_bound(z, SE, N, X2, b=1-p),
        # 'Bennett': bennett_bound(z, SE, N, X2, b=1-p),
        # 'Binom bound': binomial_bound(z*SE*N + N*p, N, p=p),
        'Binom exact': binomial_exact(z*SE*N + N*p, N, p=p),
        # 'Binom sample': binomial_sample(z*SE*N + N*p, N, p=p),
    })

    # Metadata for plotting (colors and line styles)
    plot_config = {
        'Gaussian (CLT)': {'color': 'red', 'linestyle': 'solid'},
        'Bennett': {'color': 'red', 'linestyle': 'solid'},
        'Hoeffding': {'color': 'blue', 'linestyle': 'solid'},
        'Bernstein': {'color': 'purple', 'linestyle': 'solid'},
        'Binom bound': {'color': 'green', 'linestyle': 'solid'},
        'Binom exact': {'color': 'green', 'linestyle': 'dot'},
        'Binom sample': {'color': 'green', 'linestyle': 'dash'},
    }

    metadata = {'sigma': sigma, 'N': N, 'plot_config': plot_config}

    return df, metadata


def plot_tail_bounds(N: int = 30, p: float = 0.8,
                     z_max: float = 6.0, n_points: int = 1000):
    """
    Create both regular and log-scale plots of tail bounds using Plotly.

    Args:
        N: number of samples
        p: probability parameter for binomial
        z_max: maximum z-score to plot
        n_points: number of points for plotting
    """
    # Generate data
    df, metadata = generate_iid_binomial_tailbounds(N, p, z_max, n_points)
    sigma = metadata['sigma']
    plot_config = metadata['plot_config']

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Linear Scale", "Log Scale"),
        horizontal_spacing=0.1
    )

    # Plot each bound
    for col_name in df.columns:
        if col_name == 'z':
            continue

        config = plot_config[col_name]

        # Regular scale
        fig.add_trace(go.Scatter(
            x=df['z'], y=df[col_name], name=col_name,
            line=dict(color=config['color'], width=2, dash=config['linestyle']),
            legendgroup=col_name
        ), row=1, col=1)

        # Log scale
        fig.add_trace(go.Scatter(
            x=df['z'], y=df[col_name], name=col_name,
            line=dict(color=config['color'], width=2, dash=config['linestyle']),
            legendgroup=col_name, showlegend=False
        ), row=1, col=2)

    # Update y-axis to log scale for second plot
    fig.update_yaxes(type="log", row=1, col=2)

    # Add axis labels
    fig.update_xaxes(title_text="z-score", row=1, col=1)
    fig.update_xaxes(title_text="z-score", row=1, col=2)
    fig.update_yaxes(title_text="P(Z > z)", row=1, col=1)
    fig.update_yaxes(title_text="P(Z > z)", row=1, col=2)

    fig.update_layout(
        width=1200,
        height=500,
        legend=dict(x=1.02, y=1, xanchor='left'),
        title_text=f"Tail Bounds Comparison (p={p:.2f}, N={N})"
    )

    return fig


def plot_tail_bounds_seaborn(N: int = 30, p: float = 0.8,
                              z_max: float = 6.0, n_points: int = 1000):
    """
    Create both regular and log-scale plots of tail bounds using seaborn/matplotlib.

    Args:
        N: number of samples
        p: probability parameter for binomial
        z_max: maximum z-score to plot
        n_points: number of points for plotting

    Returns:
        matplotlib figure object
    """
    # Generate data
    df, metadata = generate_iid_binomial_tailbounds(N, p, z_max, n_points)
    sigma = metadata['sigma']
    plot_config = metadata['plot_config']

    # Set seaborn style
    sns.set_style("whitegrid")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Mapping for matplotlib linestyles
    linestyle_map = {'solid': '-', 'dash': '--'}

    # Plot each bound
    for col_name in df.columns:
        if col_name == 'z':
            continue

        config = plot_config[col_name]
        linestyle = linestyle_map[config['linestyle']]

        # Regular scale
        ax1.plot(df['z'], df[col_name],
                color=config['color'],
                linestyle=linestyle,
                linewidth=2,
                label=col_name)

        # Log scale
        ax2.plot(df['z'], df[col_name],
                color=config['color'],
                linestyle=linestyle,
                linewidth=2,
                label=col_name)

    # Configure axes
    ax1.set_xlabel('z-score')
    ax1.set_ylabel('P(Z > z)')
    ax1.set_title('Regular Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('z-score')
    ax2.set_ylabel('P(Z > z)')
    ax2.set_title('Log Scale')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    fig.suptitle(f'Tail Bounds Comparison (σ={sigma:.3f}, N={N})', fontsize=14)
    plt.tight_layout()

    return fig



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot tail bounds for probability distributions')
    parser.add_argument('--N', type=int, default=100, help='Number of samples')
    parser.add_argument('--p', type=float, default=0.7, help='Probability parameter')
    parser.add_argument('--z-max', type=float, default=5, help='Maximum z-score to plot')
    parser.add_argument('--pdf', action='store_true', help='Also generate PDF output using seaborn')
    parser.add_argument('--pdf-only', action='store_true', help='Only generate PDF output (skip HTML)')

    args = parser.parse_args()

    # Save to OUTPUT directory
    import os
    os.makedirs('OUTPUT', exist_ok=True)

    # Generate plotly HTML plot (unless pdf-only)
    if not args.pdf_only:
        fig = plot_tail_bounds(N=args.N, p=args.p, z_max=args.z_max)
        fig.write_html('OUTPUT/tailbounds.html')
        print(f"Plot saved to OUTPUT/tailbounds.html")

    # Generate seaborn PDF plot if requested
    if args.pdf or args.pdf_only:
        fig_seaborn = plot_tail_bounds_seaborn(N=args.N, p=args.p, z_max=args.z_max)
        fig_seaborn.savefig('OUTPUT/tailbounds.pdf', bbox_inches='tight', dpi=300)
        plt.close(fig_seaborn)
        print(f"Plot saved to OUTPUT/tailbounds.pdf")
