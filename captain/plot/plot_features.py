import os, sys, glob
import pickle
import matplotlib.backends.backend_pdf
import matplotlib.backends.backend_svg
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
import numpy as np
from ..agents.state_monitor import FeaturesObservation
from ..biodivsim.BioDivEnv import BioDivEnv

def plot_features(env, #BioDivEnv
                  features: FeaturesObservation,
                  wd: str,
                  outfile: str,
                  protection_matrix = None,
                  quadrant_coords_list = None,
                  ):

    if env is not None:
        protection_matrix = env.bioDivGrid._protection_matrix
        quadrant_coords_list = env.quadrant_coords_list
    fig_list = []
    fig_s = [6, 5.5]
    fontsize = 15
    counter = 0
    for feature_name in features.feature_names:
        feat = features.stats_quadrant[:, counter].flatten()
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        cell_cost_matrix = np.zeros(protection_matrix.shape)
        for i in range(len(feat)):
            xy = np.meshgrid(quadrant_coords_list[i][0], quadrant_coords_list[i][1])
            cell_cost_matrix[xy[0], xy[1]] = feat[i]

        ax = sns.heatmap(
            cell_cost_matrix,
            # vmin=env._baseline_cost,
            # vmax=max_cost,
            xticklabels=False,
            yticklabels=False,
        )
        plt.gca().set_title(feature_name, fontweight="bold", fontsize=fontsize)
        fig_list.append(fig)
        counter += 1

    file_name = os.path.join(wd, os.path.basename(outfile)) + ".pdf"
    plot_biodiv = matplotlib.backends.backend_pdf.PdfPages(file_name)

    for fig in fig_list:
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        plot_biodiv.savefig(fig)
    plot_biodiv.close()
    print("Plot saved as:", file_name)

