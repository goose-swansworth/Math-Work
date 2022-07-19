import matplotlib.pyplot as plt
import numpy as np

def plot_values(plot_tups, xlabel="", ylabel="", title="",
                color=None, xlim=None, ylim=None,
                xticks=None, yticks=None):
    """General plotting function"""
    axes = plt.axes()
    for tup in plot_tups:
        x, y, label, f_str = tup
        axes.plot(x, y, f_str, label=label, color=color)
    axes.set_xlabel(xlabel, fontsize=15)
    axes.set_ylabel(ylabel, fontsize=15)
    axes.set_title(title)
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    if xticks is not None:
        ticks, labels = xticks
        axes.set_xticks(ticks)
        axes.set_xticklabels(labels)
    if yticks is not None:
        ticks, labels = yticks
        axes.set_yticks(ticks)
        axes.set_yticklabels(labels)
    plt.legend(loc="best", framealpha=0.5)
    axes.grid(True)
    return axes

def phase_plot_2d(w, w0, color=None):
    x, y = w[:, 0], w[:, 1]
    axes = plot_values([(x, y, "", "-")], xlabel="x", ylabel="y", color=color)
    return axes
    

def phase_plot_3d(w, w0, color=None):
    x, y, z = w[:, 0], w[:, 1], w[:, 2]
    axes = plt.figure().add_subplot(projection='3d')
    axes.plot(x, y, z, label=f"w0={w0}", color=color)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    plt.legend(loc="best")
    plt.show()

def time_series_plot(w, t, f_str):
    plots = []
    for i in range(w.shape[1]):
        w_i = w[:, i]
        tup = (t, w_i, f"w{i}", f_str)
        plots.append(tup)
    plot_values(plots, xlabel="t", ylabel="w")
    plt.show()
