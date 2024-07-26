import os, sys, glob
import pickle
import matplotlib.backends.backend_pdf
import matplotlib.backends.backend_svg
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=1, precision=3)  # prints floats, no scientific notation
from matplotlib.patches import Rectangle, Polygon
from matplotlib.gridspec import GridSpec
import baltic as bt
from ..biodivinit.PhyloGenerator import convert_to_bt_tree
from ..agents.state_monitor import extract_features_restore
from ..utilities.metrics import calc_STAR_from_grid, calc_MSA_from_grid

def plot_species_ranges_list(
    pklfile=None,
    loaded_env=None,
    species_list=[6, 12, 200, 350],
    log_transform=1,
    plot_titles=True,
):
    if loaded_env:
        env = loaded_env
    else:
        with open(pklfile, "rb") as pkl:
            env = pickle.load(pkl)
    evolveGrid = env.bioDivGrid
    resolution = env.resolution

    pop_sp = evolveGrid.individualsPerSpecies()
    range_sp = evolveGrid.geoRangePerSpecies()
    max_pop_sp = []
    # main_ttl = "Time: %s" % evolveGrid._counter
    abundance_map = []
    ttl = []
    titles = []
    for sp_ID in species_list:
        ttl.append(
            "Sp. %s (pop. size: %s, range size: %s)"
            % (sp_ID, round(pop_sp[sp_ID]), round(range_sp[sp_ID]))
        )
        titles.append(f"Sp. {sp_ID}")
        if log_transform:
            abundance_map.append(np.log10(1 + evolveGrid._h[sp_ID]))
            max_pop_sp.append(np.max(np.log10(1 + evolveGrid._h[sp_ID])))
        else:
            abundance_map.append(evolveGrid._h[sp_ID])
            max_pop_sp.append(np.max(evolveGrid._h[sp_ID]))

    fontsize = 15

    q_indx = env.protected_quadrants
    x_coord, y_coord = [], []
    for i in q_indx:
        x_coord.append(env.quadrant_coords_list[i][0][0])
        y_coord.append(env.quadrant_coords_list[i][1][0])

    col_outline_protected = "black"
    lwd = 1

    fig_list = []
    fig_s = [6, 5.5]

    for sp_i in range(len(species_list)):
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        mask = np.zeros(abundance_map[sp_i].shape)
        if log_transform:
            mask[abundance_map[sp_i] <= np.log10(2)] = 1
        else:
            mask[abundance_map[sp_i] < 1] = 1
        # cmap = sns.color_palette("crest")
        ax = sns.heatmap(
            abundance_map[sp_i],
            cmap="viridis_r",
            vmin=0,
            vmax=np.max(max_pop_sp),
            xticklabels=False,
            yticklabels=False,
            mask=mask,
        )
        ax.set_facecolor("#f0f0f0")
        if plot_titles:
            plt.gca().set_title(ttl[sp_i], fontweight="bold", fontsize=fontsize)
        for i in range(len(x_coord)):
            ax.add_patch(
                Rectangle(
                    (y_coord[i], x_coord[i]),
                    resolution[0],
                    resolution[1],
                    fill=False,
                    edgecolor=col_outline_protected,
                    lw=lwd,
                )
            )

        fig_list.append(fig)

    return fig_list, titles


def plot_biodiv_env(pklfile=None, loaded_env=None,
                    max_n_species=0, plot_titles=True,
                    variables=None, gray_K0=True):
    if variables is None:
        variables = ['diversity', 'density', 'rank-abundance', 'phylogeny',
                     'disturbance', 'selective-disturbance', 'climate',
                     'value', 'carbon', 'cost', 'time-series',
                     'risk-label', 'metrics'
                     ]
    if loaded_env:
        env = loaded_env
    else:
        with open(pklfile, "rb") as pkl:
            env = pickle.load(pkl)
    evolveGrid = env.bioDivGrid

    # ----
    fig_list = []
    titles = []
    fig_s = [6, 5.5]
    fontsize = 15
    col_outline_protected = "black"
    lwd = 1
    # get protected units
    q_indx = env.protected_quadrants
    x_coord, y_coord = [], []
    for i in q_indx:
        x_coord.append(env.quadrant_coords_list[i][0][0])
        y_coord.append(env.quadrant_coords_list[i][1][0])

    resolution = env.resolution
    time_series_stats = np.array(env.history) * 100
    # ----

    # plot sp richness
    if 'diversity' in variables:
        titles.append("Species richness")
        ttl = "Species richness (%s ssp.)" % evolveGrid.numberOfSpecies()
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        if not max_n_species:
            bounds = [0, np.max(evolveGrid.speciesPerCell())]
        else:
            bounds = [0, max_n_species]

        plt_var = evolveGrid.speciesPerCell()
        if np.min(evolveGrid._K_max) == 0 and gray_K0 is True:
            plt_var[evolveGrid._K_max == 0] = np.nan
        ax = sns.heatmap(
            plt_var,
            cmap="coolwarm",
            vmin=bounds[0],
            vmax=bounds[1],
            xticklabels=False,
            yticklabels=False,
        )
        if plot_titles:
            plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
        for i in range(len(x_coord)):
            ax.add_patch(
                Rectangle(
                    (y_coord[i], x_coord[i]),
                    resolution[0],
                    resolution[1],
                    fill=False,
                    edgecolor=col_outline_protected,
                    lw=lwd,
                )
            )
        fig_list.append(fig)

    if 'density' in variables:
        # population density
        titles.append("Mean population density")
        ttl = "Mean population density (mean: %s)" % round(np.mean(evolveGrid.individualsPerCell()), 1)
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        bounds = [0, np.max(evolveGrid._K_max)]
        plt_var = evolveGrid.individualsPerCell()
        if np.min(evolveGrid._K_max) == 0 and gray_K0 is True:
            plt_var[evolveGrid._K_max == 0] = np.nan
        ax = sns.heatmap(
            plt_var,
            cmap="YlGn",
            vmin=bounds[0],
            vmax=bounds[1],
            xticklabels=False,
            yticklabels=False,
        )
        if plot_titles:
            plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
        for i in range(len(x_coord)):
            ax.add_patch(
                Rectangle(
                    (y_coord[i], x_coord[i]),
                    resolution[0],
                    resolution[1],
                    fill=False,
                    edgecolor=col_outline_protected,
                    lw=lwd,
                )
            )
        fig_list.append(fig)

    if 'rank-abundance' in variables:
        # rank-abundance plot
        titles.append("Total population size")
        ttl = "Total population size (%s M)" % np.round(
            evolveGrid.numberOfIndividuals() / 1000000, 2
        )
        n_individuals_per_species = [
            np.arange(evolveGrid._n_species),
            evolveGrid.individualsPerSpecies(),
        ]
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))  # rank abundance plot
        plt.bar(
            x=n_individuals_per_species[0],
            height=n_individuals_per_species[1],
            width=0.8,
            linewidth=0,
        )
        if plot_titles:
            plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
        fig_list.append(fig)

    if 'phylogeny' in variables:
        # phylogeny
        titles.append("Phylogenetic diversity")
        pd_percentage = np.round(time_series_stats[-1, 2], 1)
        ttl = "Phylogenetic diversity (%s %%)" % pd_percentage

        extant_species = evolveGrid._all_tip_labels[evolveGrid.extantSpeciesID()]
        extinct_species = [i for i in evolveGrid._all_tip_labels if i not in extant_species]
        try:
            ll = bt.loadNewick(evolveGrid._phylo_file_name, absoluteTime=False)
            # transform species name to match ll tree
            extinct_species = [str(int(i.split("T")[1])) for i in extinct_species]
        except:
            ll = convert_to_bt_tree(evolveGrid._phylo_tree)
        grey_out = [i for i in ll.getExternal() if i.name in extinct_species]
        for tip in grey_out:
            tip.traits["inactive"] = True  ## inactivate tips
        for node in ll.getInternal():  ## iterate over internal nodes
            if len(node.leaves) == len(
                [ch for ch in node.leaves if ch in [k.name for k in grey_out]]
            ):  ## if all descendant tips are grey'd out - grey out node too
                node.traits["inactive"] = True

        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        ax = fig.add_subplot(111, facecolor="w")  # phylogeny
        colour = (
            lambda k: "red" if "inactive" in k.traits else "darkgray"
        )  ## light grey if branch has "inactive" as key in trait dict, black otherwise
        ll.plotTree(
            ax, connection_type="elbow", width=1, colour=colour
        )  ## elbow branch connection, small branch width, colour via function
        ax.set_yticks([])
        ax.set_yticklabels([])  ## remove y axis labels
        ax.set_xticks([])
        ax.set_xticklabels([])  ## remove x axis labels
        [
            ax.spines[loc].set_visible(False) for loc in ax.spines if loc not in ["bottom"]
        ]  ## remove spines
        ax.set_xlim(-0.1, ll.treeHeight + 0.1)  ## limit tree
        ax.set_ylim(-2, ll.ySpan + 2)
        if plot_titles:
            plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
        fig_list.append(fig)

    if 'disturbance' in variables:
        # disturbance
        titles.append("Disturbance")
        ttl = "Disturbance (mean: %s)" % round(
            np.mean(evolveGrid._disturbance_matrix * (1 - evolveGrid._protection_matrix)), 2
        )
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        disturbance = evolveGrid._disturbance_matrix * (1 - evolveGrid._protection_matrix)
        if np.min(evolveGrid._K_max) == 0 and gray_K0 is True:
            disturbance[evolveGrid._K_max == 0] = np.nan
        ax = sns.heatmap(
            disturbance,
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            xticklabels=False,
            yticklabels=False,
        )
        if plot_titles:
            plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
        for i in range(len(x_coord)):
            ax.add_patch(
                Rectangle(
                    (y_coord[i], x_coord[i]),
                    resolution[0],
                    resolution[1],
                    fill=False,
                    edgecolor=col_outline_protected,
                    lw=lwd,
                )
            )
        fig_list.append(fig)

    if 'selective-disturbance' in variables:
        # selective disturbance
        titles.append("Selective disturbance")
        ttl = "Selective disturbance (mean: %s)" % round(
            np.mean(
                evolveGrid._selective_disturbance_matrix
                * (1 - evolveGrid._protection_matrix)
            ),
            2,
        )
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        selective_disturbance = evolveGrid._selective_disturbance_matrix * (
            1 - evolveGrid._protection_matrix
        )
        if np.min(evolveGrid._K_max) == 0 and gray_K0 is True:
            selective_disturbance[evolveGrid._K_max == 0] = np.nan
        ax = sns.heatmap(
            selective_disturbance,
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            xticklabels=False,
            yticklabels=False,
        )
        if plot_titles:
            plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
        for i in range(len(x_coord)):
            ax.add_patch(
                Rectangle(
                    (y_coord[i], x_coord[i]),
                    resolution[0],
                    resolution[1],
                    fill=False,
                    edgecolor=col_outline_protected,
                    lw=lwd,
                )
            )
        fig_list.append(fig)

    if 'climate' in variables:
        # climate
        titles.append("Mean annual temperature")
        ttl = "Mean annual temperature (mean anomaly: %s)" % round(np.mean(evolveGrid._climate_layer), 2)
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))  # climate
        ax = sns.heatmap(
            evolveGrid._climate_layer,
            cmap="Reds",
            vmin=0,
            vmax=7,
            xticklabels=False,
            yticklabels=False,
        )
        if plot_titles:
            plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
        for i in range(len(x_coord)):
            ax.add_patch(
                Rectangle(
                    (y_coord[i], x_coord[i]),
                    resolution[0],
                    resolution[1],
                    fill=False,
                    edgecolor=col_outline_protected,
                    lw=lwd,
                )
            )
        fig_list.append(fig)

    if 'value' in variables:
        # value map
        titles.append("Economic value")
        ttl = "Economic value (%s %%)" % np.round(100 - time_series_stats[-1, 1], 1)
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))  # value map
        sp_value = evolveGrid._species_value_reference
        presence_absence = evolveGrid._h + 0
        presence_absence[
            presence_absence < 1
        ] = 0  # species_threshold is only used for total pop size
        presence_absence[presence_absence > 1] = 1  # not within each cell
        cell_value = np.log(1 + np.einsum("sij,s->ij", presence_absence, sp_value))
        bounds = [0, np.log((np.sum(sp_value)))]
        if np.min(evolveGrid._K_max) == 0 and gray_K0 is True:
            cell_value[evolveGrid._K_max == 0] = np.nan
        ax = sns.heatmap(
            cell_value, vmin=bounds[0], vmax=bounds[1], xticklabels=False, yticklabels=False
        )

        if plot_titles:
            plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
        for i in range(len(x_coord)):
            # print(i, y_coord[i], x_coord[i])
            ax.add_patch(
                Rectangle(
                    (y_coord[i], x_coord[i]),
                    resolution[0],
                    resolution[1],
                    fill=False,
                    edgecolor=col_outline_protected,
                    lw=lwd,
                )
            )
        fig_list.append(fig)

    if 'carbon' in variables:
        # carbon map
        titles.append("Carbon stored (log)")
        ttl = "Carbon stored (log): %s " % np.round(np.log(np.sum(evolveGrid.getCarbonValue_cell())), 3)
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))  # value map
        cell_value = np.log(1 + evolveGrid.getCarbonValue_cell())
        if np.min(evolveGrid._K_max) == 0 and gray_K0 is True:
            cell_value[evolveGrid._K_max == 0] = np.nan
        # bounds = [0, np.log((np.sum(sp_value)))]
        ax = sns.heatmap(
            cell_value,
            cmap="YlOrBr",
            # vmin=bounds[0], vmax=bounds[1],
            xticklabels=False, yticklabels=False
        )

        if plot_titles:
            plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
        for i in range(len(x_coord)):
            ax.add_patch(
                Rectangle(
                    (y_coord[i], x_coord[i]),
                    resolution[0],
                    resolution[1],
                    fill=False,
                    edgecolor=col_outline_protected,
                    lw=lwd,
                )
            )
        fig_list.append(fig)


    if 'cost' in variables:
        # protection cost
        titles.append("Cost of protecting")
        costs = env.getProtectCostQuadrant()
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        cell_cost_matrix = np.zeros(env.bioDivGrid._protection_matrix.shape)
        for i in range(len(costs)):
            xy = np.meshgrid(env.quadrant_coords_list[i][0], env.quadrant_coords_list[i][1])
            cell_cost_matrix[xy[0], xy[1]] = costs[i]
        ttl = "Cost of protecting (mean: %s)" % round(
            np.mean(costs + env._baseline_cost), 2
        )

        max_cost = np.max([env._baseline_cost + env._cost_coeff * np.prod(env.resolution),
                           np.max(cell_cost_matrix)])
        if np.min(evolveGrid._K_max) == 0 and gray_K0 is True:
            cell_cost_matrix[evolveGrid._K_max == 0] = np.nan
        ax = sns.heatmap(
            cell_cost_matrix,
            vmin=env._baseline_cost,
            vmax=max_cost,
            xticklabels=False,
            yticklabels=False,
        )
        if plot_titles:
            plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
        for i in range(len(x_coord)):
            ax.add_patch(
                Rectangle(
                    (y_coord[i], x_coord[i]),
                    resolution[0],
                    resolution[1],
                    fill=False,
                    edgecolor=col_outline_protected,
                    lw=lwd,
                )
            )
        fig_list.append(fig)

    if 'time-series' in variables:

        # calculate number of species at risk
        weights = np.array([12, 3, 1, 1, 2.])
        rl_lab = ['CR', 'EN', 'VU', 'NT', 'LC']
        status_indx = [i for i in range(len(env.history_var_names)) if env.history_var_names[i] in  rl_lab]
        status = np.array(env.history)[:, np.array(status_indx)]
        threatened_species = np.einsum('ts -> t', status[:,:3]) * env.n_species
        not_threatened_species = np.einsum('ts -> t', status[:, 3:]) * env.n_species
        # print("status", status)
        # show as percentage change from initial state
        threatened_species = 100 * threatened_species / np.max([1, threatened_species[0]]) - 100
        not_threatened_species = 100 * not_threatened_species / np.max([1, not_threatened_species[0]]) - 100
        # print("not_threatened_species", not_threatened_species)
        # print("threatened_species", threatened_species)
        ch = np.array(env.history)[:, 3]
        # i_tmp = env.grid_obj_previous.getCarbonValue_cell() > 0
        # carb_den = np.max(
        #     env.grid_obj_previous.getCarbonValue_cell()[i_tmp])
        # carb_time_series = 100 * ((ch * env._init_total_carbon - env._init_total_carbon) / (carb_den))
        carb_time_series = 100 * np.array(env.history)[:, 3] - 100
        rel_pop_size = 100 * np.array(env.history)[:, 4] - 100


        # diversity/PD/value trajectories
        titles.append("Variables through time")
        ttl = "Variables through time"
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        x = np.arange(len(time_series_stats))
        # plt.plot(x, time_series_stats[:, 1], "o-", color="#7570b3")  # value
        # plt.plot(x, time_series_stats[:, 2], "o-", color="#1b9e77")  # PD
        # plt.plot(x, time_series_stats[:, 0], "o-", color="#d95f02")  # sp num
        plt.plot(x, not_threatened_species, "o-", color="#1b9e77")  # not threatened species
        plt.plot(x, threatened_species, "o-", color="#d95f02")  # threatened species
        plt.plot(x, carb_time_series, "o-", color="b")# TotalCarbon
        plt.plot(x, rel_pop_size, "o-", color="#bcbddc")  # rel_pop_size
        if plot_titles:
            plt.gca().set_title(
                "Variables through time", fontweight="bold", fontsize=fontsize
            )
        plt.legend(
            labels=(
                # "Economic value",
                # "Phylogenetic diversity",
                # "Species diversity",
                "Not threatened species",
                "Threatened species",
                "Total carbon",
                "rel_pop_size"),
            # loc="center left", # "lower left"
            facecolor='white'
        )
        if len(time_series_stats) < 31:
            x_max = 31
        else:
            x_max = len(time_series_stats) + 1

        plotted_vars = np.array([carb_time_series, not_threatened_species, threatened_species])
        if np.min(plotted_vars) < 90:
            y_min = np.min(plotted_vars) - 5
        else:
            y_min = 90
        if np.max(plotted_vars) == 100:
            y_max = 116
        else:
            y_max = np.max(plotted_vars) + 5

        # y_min, y_max = 85, 120


        plt.axis([-1, x_max, y_min, y_max])  # TODO: expose xlim and ylim
        plt.xlabel("Time")
        plt.ylabel("Relative change")
        fig_list.append(fig)

    if 'risk-label' in variables:
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        labels, label_count = np.unique(env._init_sp_ext_risk, return_counts=True)
        # counts based on current assessment
        # lab, count_protected = np.unique(env.getExtinction_risk_labels()[f.protected_species_list],
        #                                  return_counts=True)
        # risk_labels = env.getExtinction_risk_labels()
        # labels, label_count = np.unique(risk_labels, return_counts=True)
        risk_labels = env.getExtinction_risk_labels()
        titles.append("Initial species extinction risk (%s sp)" % np.sum(env._init_sp_ext_risk < 3))
        ttl = "Initial species extinction risk (%s sp)" % np.sum(env._init_sp_ext_risk < 3)

        for i in env.species_risk_criteria.available_labels:
            if i not in labels:
                label_count = np.insert(label_count, i, 0)

        readable_labels = env.species_risk_criteria.available_labels_text
        p = sns.color_palette("Spectral")
        ax = sns.barplot(x=readable_labels, y=label_count, hue=readable_labels,
                    palette=p, linewidth=1, edgecolor="k")

        risk_labels_2 = env.getExtinction_risk_labels()
        _, label_count_2 = np.unique(risk_labels_2, return_counts=True)

        max_y = 1.05 * np.max([np.max(label_count), np.max(label_count_2)])
        ax.set(ylim=(0, max_y))

        if plot_titles:
            plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
        fig_list.append(fig)


        # extinction risk labels
        risk_labels = env.getExtinction_risk_labels()
        titles.append("Current species extinction risk (%s sp)" % np.sum(risk_labels < 3))
        ttl = "Current species extinction risk (%s sp)" % np.sum(risk_labels < 3)
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        labels, label_count = np.unique(risk_labels, return_counts=True)
        for i in range(5):
            if i not in labels:
                label_count = np.insert(label_count, i, 0)

        readable_labels = env.species_risk_criteria.available_labels_text
        p = sns.color_palette("Spectral")
        ax = sns.barplot(x=readable_labels, y=label_count, hue=readable_labels,
                    palette=p, linewidth=1, edgecolor="k")
        ax.set(ylim=(0, max_y))

        if plot_titles:
            plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
        fig_list.append(fig)

        # risk trajectories
        titles.append("Risk through time")
        ttl = "Risk through time"
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        x = np.arange(len(time_series_stats))
        p = sns.color_palette("Spectral", n_colors=5)
        indx_0 = 5
        plt.plot(x, time_series_stats[:, indx_0+0], "o-", color=p[0])
        plt.plot(x, time_series_stats[:, indx_0+1], "o-", color=p[1])
        plt.plot(x, time_series_stats[:, indx_0+2], "o-", color=p[2])
        plt.plot(x, time_series_stats[:, indx_0+3], "o-", color=p[3])
        plt.plot(x, time_series_stats[:, indx_0+4], "o-", color=p[4])
        if plot_titles:
            plt.gca().set_title(
                "Species conservation status", fontweight="bold", fontsize=fontsize
            )
        plt.legend(
            labels=readable_labels,
            # loc="center left", # "lower left"
            facecolor='white'
        )
        if len(time_series_stats) < 31:
            x_max = 31
        else:
            x_max = len(time_series_stats) + 1
        if np.min(time_series_stats[:,4:]) < 80:
            y_min = np.min(time_series_stats[:,4:]) - 5
        else:
            y_min = 80
        if np.max(time_series_stats[:,4:]) == 100:
            y_max = 105
        else:
            y_max = np.max(time_series_stats[:,4:]) + 5

        plt.axis([-1, x_max, y_min, y_max])  # TODO: expose xlim and ylim
        plt.xlabel("Time")
        plt.ylabel("Percentage values")
        fig_list.append(fig)

    if 'protected-threatened' in variables:
        f = extract_features_restore(grid_obj=env.grid_obj_most_recent,
                                        grid_obj_previous=env.grid_obj_previous,
                                        quadrant_resolution=env.resolution,
                                        current_protection_matrix=env.bioDivGrid.protection_matrix,
                                        species_threat_label=env.getExtinction_risk_labels(),
                                        n_threat_labels=env.species_risk_criteria.n_labels,
                                        quandrant_grid_indx=env._quandrant_grid_indx,
                                        cost_quadrant=env.getProtectCostQuadrant(),
                                        budget=env.budget,
                                        normalize=True,
                                        get_protected_species_list=True
                                        )
        if len(f.protected_species_list):
            # counts based on initial threat assessemnt
            lab, count_protected = np.unique(env._init_sp_ext_risk[f.protected_species_list],
                                             return_counts=True)

            labels, label_count = np.unique(env._init_sp_ext_risk, return_counts=True)
            # counts based on current assessment
            # lab, count_protected = np.unique(env.getExtinction_risk_labels()[f.protected_species_list],
            #                                  return_counts=True)
            # risk_labels = env.getExtinction_risk_labels()
            # labels, label_count = np.unique(risk_labels, return_counts=True)

            for i in env.species_risk_criteria.available_labels:
                if i not in labels:
                    label_count = np.insert(label_count, i, 1) # set to 1 to avoid divide by 0


            for i in env.species_risk_criteria.available_labels:
                if i not in lab:
                    count_protected = np.insert(count_protected, i, 0)

            fr = np.round(len(f.protected_species_list) / env.bioDivGrid._n_species, 2)
            titles.append("Fraction of protected species (%s)" % fr)
            ttl = "Fraction of protected species (%s)" % fr
            fig = plt.figure(figsize=(fig_s[0], fig_s[1]))

            readable_labels = env.species_risk_criteria.available_labels_text
            p = sns.color_palette("Spectral")
            sns.barplot(x=readable_labels, y=count_protected / label_count,
                        palette=p, linewidth=1, edgecolor="k")

            if plot_titles:
                plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
            fig_list.append(fig)


            titles.append("Number of species in protected areas (%s)" % len(f.protected_species_list))
            ttl = "Number of species in protected areas (%s)" % len(f.protected_species_list)
            fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
            readable_labels = env.species_risk_criteria.available_labels_text
            p = sns.color_palette("Spectral")
            ax = sns.barplot(x=readable_labels, y=label_count,
                         facecolor="k", alpha=0.2, linewidth=1, edgecolor="k")
            sns.barplot(x=readable_labels, y=count_protected,
                        palette=p, linewidth=1, edgecolor="k")

            # plot based on current status (to set y axis)
            risk_labels_2 = env.getExtinction_risk_labels()
            _, label_count_2 = np.unique(risk_labels_2, return_counts=True)


            max_y = 1.05 * np.max([np.max(label_count), np.max(label_count_2)])

            ax.set(ylim=(0, max_y))
            if plot_titles:
                plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
            fig_list.append(fig)


            # plot based on current status
            lab, count_protected = np.unique(env.getExtinction_risk_labels()[f.protected_species_list],
                                             return_counts=True)
            risk_labels = env.getExtinction_risk_labels()
            labels, label_count = np.unique(risk_labels, return_counts=True)

            for i in env.species_risk_criteria.available_labels:
                if i not in labels:
                    label_count = np.insert(label_count, i, 1) # set to 1 to avoid divide by 0

            for i in env.species_risk_criteria.available_labels:
                if i not in lab:
                    count_protected = np.insert(count_protected, i, 0)


            titles.append("Current number of species in protected areas (%s)" % len(f.protected_species_list))
            ttl = "Current number of species in protected areas (%s)" % len(f.protected_species_list)
            fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
            readable_labels = env.species_risk_criteria.available_labels_text
            p = sns.color_palette("Spectral")
            ax = sns.barplot(x=readable_labels, y=label_count,
                         facecolor="k", alpha=0.2, linewidth=1, edgecolor="k")
            sns.barplot(x=readable_labels, y=count_protected,
                        palette=p, linewidth=1, edgecolor="k")
            ax.set(ylim=(0, max_y))
            if plot_titles:
                plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
            fig_list.append(fig)

    if 'metrics' in variables:
        try:
            star_t, star_r = calc_STAR_from_grid(env.grid_obj_previous.h, env.grid_obj_most_recent.h,
                                          quandrant_grid_indx=env._quandrant_grid_indx,
                                          sp_natural_range=env.grid_obj_previous.geoRangePerSpecies(),
                                          sp_ext_risk=env.getExtinction_risk_labels())

            msa = calc_MSA_from_grid(env.grid_obj_previous.h, env.grid_obj_most_recent.h,
                                     quandrant_grid_indx=env._quandrant_grid_indx)


            s = int(np.sqrt(star_t.size))
            star_t = star_t.reshape((s,s))
            star_r = star_r.reshape((s,s))
            msa = msa.reshape((s,s))

            # PLOT METRICS PER QUADRANT
            ttl = "STAR_t"
            titles.append(ttl)
            fig = plt.figure(figsize=(fig_s[0], fig_s[1]))  # value map
            cell_value = star_t
            if np.min(evolveGrid._K_max) == 0 and gray_K0 is True:
                cell_value[evolveGrid._K_max == 0] = np.nan
            ax = sns.heatmap(cell_value, xticklabels=False, yticklabels=False)

            if plot_titles:
                plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
            for i in range(len(x_coord)):
                ax.add_patch(
                    Rectangle(
                        (y_coord[i], x_coord[i]),
                        resolution[0],
                        resolution[1],
                        fill=False,
                        edgecolor=col_outline_protected,
                        lw=lwd,
                    ))
            fig_list.append(fig)

            ttl = "STAR_r"
            titles.append(ttl)
            fig = plt.figure(figsize=(fig_s[0], fig_s[1]))  # value map
            cell_value = star_r
            if np.min(evolveGrid._K_max) == 0 and gray_K0 is True:
                cell_value[evolveGrid._K_max == 0] = np.nan
            ax = sns.heatmap(cell_value, xticklabels=False, yticklabels=False)

            if plot_titles:
                plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
            for i in range(len(x_coord)):
                ax.add_patch(
                    Rectangle(
                        (y_coord[i], x_coord[i]),
                        resolution[0],
                        resolution[1],
                        fill=False,
                        edgecolor=col_outline_protected,
                        lw=lwd,
                    ))
            fig_list.append(fig)

            ttl = "MSA"
            titles.append(ttl)
            fig = plt.figure(figsize=(fig_s[0], fig_s[1]))  # value map
            cell_value = msa
            if np.min(evolveGrid._K_max) == 0 and gray_K0 is True:
                cell_value[evolveGrid._K_max == 0] = np.nan
            ax = sns.heatmap(cell_value, xticklabels=False, yticklabels=False)

            if plot_titles:
                plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
            for i in range(len(x_coord)):
                ax.add_patch(
                    Rectangle(
                        (y_coord[i], x_coord[i]),
                        resolution[0],
                        resolution[1],
                        fill=False,
                        edgecolor=col_outline_protected,
                        lw=lwd,
                    ))
            fig_list.append(fig)
        except:
            pass

        # PLOT METRICS HISTORIES
        msa = np.array(env.history)[:, np.where(env.history_var_names=='MSA')[0]]

        # diversity/PD/value trajectories
        titles.append("Metrics through time")
        ttl = "Metrics through time"
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        x = np.arange(len(time_series_stats))
        plt.plot(x, msa, "o-", color="#1b9e77")  # msa
        if plot_titles:
            plt.gca().set_title(
               ttl, fontweight="bold", fontsize=fontsize
            )
        plt.legend(
            labels=(
                "MSA"),
            # loc="center left", # "lower left"
            facecolor='white'
        )
        if len(time_series_stats) < 31:
            x_max = 31
        else:
            x_max = len(time_series_stats) + 1
        y_min = 0
        y_max = 1
        plt.axis([-1, x_max, y_min, y_max])  # TODO: expose xlim and ylim
        plt.xlabel("Time")
        plt.ylabel("Metrics value")
        fig_list.append(fig)

    return fig_list, titles


def _plot_env_state_init(env, species_list=None, plot_titles=True, variables=None):
    if species_list == None:
        species_list = range(env.n_species)

    fig_list, titles = plot_biodiv_env(plot_titles=plot_titles,
                                       loaded_env=env,
                                       variables=variables)

    if len(species_list):
        # fig_list = fig_list + plot_species_ranges_list(
        species_figs, species_titles = plot_species_ranges_list(
            loaded_env=env,
            species_list=species_list,
            log_transform=1,
            plot_titles=plot_titles,
        )
        fig_list.extend(species_figs)
        titles.extend(species_titles)

    return fig_list, titles


def _plot_env_state_plot_fig(fig, outfile_name, wd, plot_count, title, file_format):
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    file_name = (
        os.path.join(wd, os.path.basename(outfile_name))
        + f"_p{plot_count} - {title}.{file_format}"
    )

    print(f"Save fig '{file_name}'")
    fig.savefig(file_name)
    return file_name


def plot_env_state(
        env,
        wd=".",
        outfile="sim",
        species_list=None,
        plot_titles=True,
        file_format="one_pdf",
        variables=None,
):
    fig_list, titles = _plot_env_state_init(
        env=env, species_list=species_list, plot_titles=plot_titles,
        variables=variables,
    )

    outfile_name = "%s_step_%s" % (outfile, env.currentIteration)

    if file_format == "one_pdf":
        file_name = os.path.join(wd, os.path.basename(outfile_name)) + ".pdf"
        plot_biodiv = matplotlib.backends.backend_pdf.PdfPages(file_name)

        for fig in fig_list:
            fig.tight_layout()
            fig.subplots_adjust(top=0.92)
            plot_biodiv.savefig(fig)
        plot_biodiv.close()
        print("Plot saved as:", file_name)
        return None

    if file_format not in ["pdf", "svg", "png", "jpg"]:
        file_format = "pdf"

    plot_count = 0
    for fig, title in zip(fig_list, titles):
        _plot_env_state_plot_fig(
            fig=fig,
            outfile_name=outfile_name,
            wd=wd,
            plot_count=plot_count,
            title=title,
            file_format=file_format,
        )
        plot_count += 1


def plot_env_state_generator(
    env,
    wd=".",
    outfile="sim",
    species_list=None,
    variables=None
):
    fig_list, titles = _plot_env_state_init(
        env=env, species_list=species_list, plot_titles=True,
        variables=variables,
    )

    outfile_name = "%s_step_%s" % (outfile, env.currentIteration)

    file_format = "svg"

    plot_count = 0
    for fig, title in zip(fig_list, titles):
        filename = _plot_env_state_plot_fig(
            fig=fig,
            outfile_name=outfile_name,
            wd=wd,
            plot_count=plot_count,
            title=title,
            file_format=file_format,
        )

        yield {
            "type": "plot",
            "status": "progress",
            "data": {
                "step": env.currentIteration,
                "plot": plot_count,
                "num_plots": len(fig_list),
                "filename": filename,
                "title": title,
            },
        }

        plot_count += 1
