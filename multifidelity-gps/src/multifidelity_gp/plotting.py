"""Plot styling utilities."""

import matplotlib.pyplot as plt


def configure_plotting() -> None:
    """Apply the same publication-style plotting parameters used in the original scripts."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.fontsize"] = 9
