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
import glob
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def find_indices(l: list, tbl: pd.DataFrame):
    indices = []
    for i in l:
        indices.append(np.where(tbl.columns == i)[0][0])
    return np.array(indices)

def plot_summary_res(res,
                     par,
                     reference_var='random',
                     plot_abs_values=False,
                     outliers=True,
                     title=None,
                     show=True):
    indx = find_indices([par], res['random'])
    res_subset = []
    colnames = []
    for k in res.keys():
        r = res[k].to_numpy()
        res_subset = res_subset + list(r[:, indx].flatten())
        colnames = colnames + [k for _ in range(len(r[:, indx]))]

    tbl = np.array([res_subset, colnames]).T
    res_pd = pd.DataFrame(list(tbl))
    res_pd.columns = [par, "Policy"]
    res_pd = res_pd.astype({par: 'float'})  # .dtypes

    # reshape for rescaling
    tbl = np.array([res_subset]).reshape((len(res), 100))
    reference = np.where(np.array(list(res.keys())) == reference_var)[0][0]
    tbl_r = (tbl / (tbl[reference, :]) - 1) * 100
    np.mean(tbl_r, 1)

    tbl = np.array([list(tbl_r.flatten()), colnames]).T
    res_pd_r = pd.DataFrame(tbl)
    par_r = "Change relative to '" + reference_var + "' policy (%)"
    res_pd_r.columns = [par_r, "Policy"]
    res_pd_r = res_pd_r.astype({par_r: 'float'})  # .dtypes

    # plot
    sns.set(font_scale=1.4)
    if plot_abs_values:
        fig = plt.figure(figsize=(20, 10))
        fig.add_subplot(1, 2, 1)
    else:
        fig = plt.figure(figsize=(15, 10))
    plt.axvline(x=0, color='#636363', linestyle='--')
    _ = sns.boxplot(data=res_pd_r,
                     y='Policy', x=par_r,
                     showfliers=outliers)
    plt.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.1)
    if title is not None:
        plt.gca().set_title(title, fontweight="bold", fontsize=24)

    if plot_abs_values:
        fig.add_subplot(1, 2, 2)
        sns.set(font_scale=1.4)
        ax = sns.violinplot(data=res_pd, y='Policy', x=par, width=0.8)
        ax.set(ylabel=None)
        ax.set(yticklabels=[])  # remove the tick labels
        ax.tick_params(left=False)  # remove the ticks
        # plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1)

    if show:
        plt.show()
    else:
        return fig


def plot_summary_res_bubble(res,
                            var_x="Cost",
                            var_y = "MSA",
                            var_z = "CR",
                            reference_var='random',
                            title=None,
                            show=True,
                            decimals=1,
                            plot_mean = False,
                            seaborn_plot=False,
                            dynamic_plot = False,
                            return_res_tbl=False
                            ):
    summary_res_col = list(res['random'].columns)
    summary_res = []
    summary_res_all = []
    summary_res_p_val = []

    for par in summary_res_col:
        indx = find_indices([par], res['random'])
        res_subset = []
        colnames = []
        colnames_all = []
        for k in res.keys():
            r = res[k].to_numpy()
            res_subset = res_subset + list(r[:, indx].flatten())
            colnames.append(k)
            colnames_all = colnames_all + [k for _ in range(len(r[:, indx]))]

        # reshape for rescaling
        tbl = np.array([res_subset]).reshape((len(res), 100))
        reference = np.where(np.array(list(res.keys())) == reference_var)[0][0]
        tmp = tbl[reference, :]
        tmp[tmp == 0] = 1
        tbl_r = (tbl / tmp -1) * 100
        mean_rel_change = np.round(np.nanmean(tbl_r, 1), decimals)
        # print(par, mean_rel_change)
        summary_res.append(list(mean_rel_change))
        summary_res_all.append(list(tbl_r.flatten()))
        #summary_res_p_val.append()

    all_res = np.array(summary_res_all).T
    all_res_model_names = np.hstack((np.array(colnames_all).reshape(len(colnames_all), 1),
                                     all_res))
    a = all_res_model_names.astype('O')
    a[:, 1:] = a[:, 1:].astype(float)


    pd_all = pd.DataFrame(a, columns=["Simulated policy"] + summary_res_col)
    pd_all[var_z] = pd_all[var_z] + abs(np.min(pd_all[var_z]))
    pd_all[var_z + "rescale"] = pd_all[var_z] / (np.max(pd_all[var_z]) - np.min(pd_all[var_z])) + 0.1

    pd_res = pd.DataFrame(np.array(summary_res).T, columns=summary_res_col)
    pd_res['Simulated policy'] = colnames
    # print(pd_res[var_z] / np.mean(pd_res[var_z]) - np.mean(pd_res[var_z]))
    pd_res[var_z] = np.round(pd_res[var_z] + abs(np.min(pd_res[var_z])), decimals)
    pd_res[var_z + "rescale"] = np.round(pd_res[var_z] / (np.max(pd_res[var_z]) - np.min(pd_res[var_z])), 2) + 0.1
    # pd_res[var_z] = np.round(pd_res[var_z] / np.mean(pd_res[var_z]) - np.mean(pd_res[var_z]), 2)
    # print("pd_res", pd_res.columns)

    if return_res_tbl:
        return pd_res, pd_all

    if seaborn_plot:
        sns.set_style("darkgrid")
        fig = plt.figure(figsize=(12, 8))
        # fig.add_subplot(1, 2, 1)
        plt.gca().set_title("test", fontweight="bold", fontsize=24)
        sns.scatterplot(data=pd_res,
                        x=var_x,
                        y=var_y,
                        size=var_z,
                        hue="sim",
                        alpha=1,
                        legend=True,
                        sizes=(100, 5000))

        # plt.xlabel("Gdp per Capita")
        # plt.ylabel("Life Expectancy")
        dy = np.max(pd_res[var_y]) - np.min(pd_res[var_y])
        plt.ylim([np.min(pd_res[var_y]) - abs(0.2 * dy),
                  np.max(pd_res[var_y]) + abs(0.2 * dy) ])

        dx = np.max(pd_res[var_x]) - np.min(pd_res[var_x])
        plt.xlim([np.min(pd_res[var_x]) - abs(0.2 * dx),
                  np.max(pd_res[var_x]) + abs(0.2 * dx) ])

        # Locate the legend outside of the plot
        # fig.add_subplot(1, 2, 2)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=17)
    else:
        title = "Percentage change in '%s' (x axis), '%s' (y axis), and '%s' (size of the circles) relative to '%s' policy" % (
            var_x, var_y, var_z, reference_var
        )
        if plot_mean is False:
            fig = px.scatter(pd_all,
                             x=var_x, y=var_y,
                             size=pd_all[var_z + "rescale"].astype(float),
                             color="Simulated policy",
                             log_x=False, size_max=20, hover_name="Simulated policy",
                             hover_data=[var_z],
                             title=title)
        else:
            fig = px.scatter(pd_res,
                             x=var_x, y=var_y,
                             size=var_z + "rescale",
                             color="Simulated policy",
                             log_x=False, size_max=60, hover_name="Simulated policy",
                             hover_data=[var_z],
                             title=title)


        # Create figure and add one scatter trace
        if dynamic_plot:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=pd_res[var_x], y=pd_res[var_y],
                visible=True,
                mode='markers',
                marker=dict(color=np.arange(5),
                            # colorscale='aquamarine',
                            opacity=1, size=pd_res[var_z + "rescale"],
                            # sizemode='area', sizeref=sizeref,
                            #
                            sizemin=20, showscale=True
                            )))

            fig.add_annotation(dict(text="",
                                    showarrow=False,
                                    yref='paper', xref='paper',
                                    x=0.99, y=0.95))

             # Create x and y buttons
            x_buttons = []
            y_buttons = []
            var_y_list = ["MSA", "PDF", "STAR-t", "STAR-r", "CR", "CR_pr", "EN", "EN_pr"]

            for column in var_y_list:
                x_buttons.append(dict(method='update',
                                      label=column,
                                      args=[{'x': [pd_res[column]]}]
                                      )
                                 )

                y_buttons.append(dict(method='update',
                                      label=column,
                                      args=[{'y': [pd_res[column]]}]
                                      )
                                 )

            # Pass buttons to the updatemenus argument
            fig.update_layout(updatemenus=[dict(buttons=x_buttons, direction='up', x=0.5, y=-0.1),
                                           dict(buttons=y_buttons, direction='right', x=-0.01, y=0.5)])

            with open('p_graph.html', 'w') as f:
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

            f.close()

    if show:
        fig.show()
    else:
        return fig









