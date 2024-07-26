import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

### PARSE OUTPUT
output_name = "test_protect.pkl"
pklfile = os.path.join(wd, output_name)

pklfile = "/Users/dsilvestro/Software/captain-dev/trained_models/outputF.pkl_4322.pkl"
with open(pklfile, "rb") as pkl:
    env = pickle.load(pkl)

protected_units = np.sum(env.bioDivGrid.protection_matrix)

target_met = env.bioDivGrid.protectedIndPerSpecies() / env.min_pop_requirement - 1

# initial target
min_pop_requirement = env.bioDivGrid.individualsPerSpecies() * env.protect_fraction


n_protected_species = len(target_met[target_met > 0])
non_protected_species = env.bioDivGrid._n_species - n_protected_species
used_budget = (env._initialBudget - env.budget) / env._initialBudget
print(protected_units, "protected units", "used budget (%): ", used_budget)
print(non_protected_species, "non protected species")

cpt_protection = env.bioDivGrid.protection_matrix


# PLOT - histogram
import matplotlib
import matplotlib.pyplot as plt

target_fraction = env.protect_fraction
bins = np.linspace(0, 1, 51)  # range(11) #[0,1, 2, 5, 10]
col = ["#ef8a62", "#67a9cf"]  # ["#bd0026", "#bdc9e1"]
colors = [col[0]] * len(bins[bins < target_fraction]) + [col[1]] * len(
    bins[bins > target_fraction]
)
edgecolor = colors  # "#252525"#
Ylim = 55563
protected_fraction = (
    env.bioDivGrid.protectedIndPerSpecies() / env.bioDivGrid.individualsPerSpecies()
)
fig = plt.figure(figsize=(8, 8))
h = np.histogram(protected_fraction, bins=bins)
den = len(protected_fraction)
plt.bar(
    height=(h[0] / den) * 100,
    x=(h[1] + h[1][1] / 2)[:-1],
    color=colors,
    width=bins[1],
    log=False,
    edgecolor=edgecolor,
    linewidth=2,
)
# plt.yticks(ticks=([0, 10, 20, 30, 40])) #, labels=[5, 10, 30])
plt.gca().set_title("CAPTAIN protection outcome", fontweight="bold", fontsize=12)
plt.ylabel("Percentage of species")
plt.xlabel("Fraction of protected range")
plt.axvline(x=target_fraction, linewidth=2, color="r", linestyle="--")
fig.show()

# PLOT - map
# -- coordinates for plotting
coords = env.bioDivGrid.coords
if coords is None:
    raise FileExistsError("Coordinates not found in env")
lat_coords = np.array(coords["Coord_y"])
lon_coords = np.array(coords["Coord_x"])


# TODO additional plots:
"""
1. species range plots (highlighting protected areas)
2. scatter plot pop size vs fraction protected
"""



# COLOR MAP
# summarize across reps
def plot_conservation_map(env_list, pkl_files=None):
    if pkl_files:
        # create env list from files
        env_list = [cn.load_pickle_file(f) for f in pkl_files]
    
    avg_mapped_freq_protection = np.empty(0)
    n_reps = len(env_list)
    for env in env_list:
        cpt_protection = env.bioDivGrid.protection_matrix
        # -- coordinates for plotting
        coords = env.bioDivGrid.coords
        if coords is None:
            raise FileExistsError("Coordinates not found in env")
        lat_coords = np.array(coords["Coord_y"])
        lon_coords = np.array(coords["Coord_x"])
        
        cmap = "viridis"
        # cmap = plt.get_cmap('viridis', 10)
        min_freq = 5  # 5 # below 5% plot in grey
        # ---- plot CAPTAIN results
        freq_protection = cpt_protection / n_reps * 100
        mapped_freq_protection = np.zeros(coords["PUID"].shape)
        for i in range(len(lat_coords)):
            indx = np.where(env.bioDivGrid._pus_id == coords["PUID"][i])[0]
            if len(indx):  # if cell in data
                mapped_freq_protection[i] = freq_protection[indx]
        
        if avg_mapped_freq_protection.size:
            avg_mapped_freq_protection += mapped_freq_protection
        else:
            avg_mapped_freq_protection = mapped_freq_protection

    avg_mapped_freq_protection = np.round(avg_mapped_freq_protection).astype(int)
    fig = plt.figure(figsize=(4, 8))
    scatter = plt.scatter(
        lon_coords[avg_mapped_freq_protection <= min_freq],
        lat_coords[avg_mapped_freq_protection <= min_freq],
        c="#bdbdbd",
        marker=",",
        s=2,
    )
    scatter = plt.scatter(
        lon_coords[avg_mapped_freq_protection > min_freq],
        lat_coords[avg_mapped_freq_protection > min_freq],
        c=avg_mapped_freq_protection[avg_mapped_freq_protection > min_freq],
        marker=",",
        s=2,
        cmap=cmap,
    )
    title = "CAPTAIN solution"
    plt.gca().set_title(title, fontweight="bold", fontsize=12)
    leg = scatter.legend_elements(num=[0, 25, 50, 75, 100])
    leg = (leg[0][::-1], leg[1][::-1])
    plt.legend(*leg, loc="upper left", title="Ranking (%)", facecolor="white")
    
    # file_name = os.path.join(wd, outfile + "_map.pdf")
    # plot_biodiv = matplotlib.backends.backend_pdf.PdfPages(file_name)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)  #
    fig.show()


















