import os
import glob
import numpy as np
np.set_printoptions(suppress=True, precision=3)
import rasterio
import captain as cn
import seaborn as sns
import matplotlib.pyplot as plt
from captain.utilities import empirical_data_parser as cn_util
from captain.utilities import sdm_utils as sdm_utils
from captain.biodivsim import SimGrid as cn_sim_grid
import rioxarray as rxr
import sparse

data_wd = "/Users/dsilvestro/Documents/Projects/Ongoing/Captain_NZ/data"
env_data_wd = os.path.join(data_wd, "Disturbance and cost")
sdm_data_wd = os.path.join(data_wd, "Present SDMs")
results_wd = os.path.join(data_wd, "test_res")

max_disturbance = 0.75
min_disturbance = 0.1
max_species_cuptoff = 20 #None
do_plots = True

# parse SDM files and create 3D object
sdm_files = np.sort(glob.glob(sdm_data_wd + "/*.tif"))
sdms = []
sp_names = []
for i in range(len(sdm_files)):
    sp_data, taxon_name = sdm_utils.get_sdm_data(sdm_files, i, cropped=None)
    sdms.append(sp_data)
    sp_names.append(taxon_name)
    if i + 1 == max_species_cuptoff:
        break

sdms = np.squeeze(np.array(sdms)) # shape = (species, lon, lat)

prob_threshold = 0.5
suitability = cn_util.get_habitat_suitability(sdms, prob_threshold)
species_richness = np.nansum(suitability, axis=0)
original_grid_shape = species_richness.shape

if do_plots:
    sns.heatmap(species_richness, cmap="YlGnBu")
    plt.title("Species richness")
    plt.show()

# MAKE IT A SQUARE without gaps
reference_grid_pu = species_richness > 0
n_pus = reference_grid_pu[reference_grid_pu > 0].size
n_species = sdms.shape[0]
# could also further reduce the grid by removing areas with few species

# sns.heatmap(reference_grid_pu)
# plt.title("Reference grid")
# plt.show()


grid_length = np.round(np.sqrt(n_pus)).astype(int) + 1
if grid_length % 2 != 0:
    # make it a multiple of 2
    grid_length += 1

# reduce SDMs
graph_sdms = []
for scaled_dd in sdms:
    prep_vec = np.zeros(grid_length ** 2)
    species_vec = scaled_dd[reference_grid_pu > 0].flatten()
    prep_vec[:n_pus] += species_vec
    species_grid = prep_vec.reshape((grid_length, grid_length))
    species_grid[np.isnan(species_grid)] = 0
    graph_sdms.append(species_grid)

graph_sdms = np.array(graph_sdms)
graph_suitability = cn_util.get_habitat_suitability(graph_sdms, prob_threshold)

if do_plots:
    # plot one graph SDM
    indx = 1
    sns.heatmap(graph_sdms[indx], cmap="YlGnBu")
    plt.title("Graph - SDM")
    plt.show()
    # backtransform
    tmp = graph_sdms[indx].flatten()[:-(graph_sdms[indx].size - n_pus)] + 0
    z = np.zeros(original_grid_shape)
    z[reference_grid_pu > 0] += tmp
    np.all(z == sdms[indx])
    sns.heatmap(z, cmap="YlGn")
    plt.show()

# reduce coordinates
xy_coords = np.meshgrid(np.arange(original_grid_shape[1]), np.arange(original_grid_shape[0]))
lon = xy_coords[0]
lon_vec = np.zeros(grid_length ** 2)
lon_vec[:n_pus] += lon[reference_grid_pu > 0].flatten()
lon_red = lon_vec.reshape((grid_length, grid_length))

lat = xy_coords[1]
lat_vec = np.zeros(grid_length ** 2)
lat_vec[:n_pus] += lat[reference_grid_pu > 0].flatten()
lat_red = lat_vec.reshape((grid_length, grid_length))

# dispersal threshold (in number of cells)
disp_threshold = 3
fname_sparse = "disp_probs_sp%s_th%s.npz" % (n_species, disp_threshold)
try:
    dispersal_probs_sparse = sparse.load_npz(os.path.join(data_wd, fname_sparse))
except:
    dispersal_probs = cn_sim_grid.dispersalDistancesThresholdCoord(grid_length,
                                                                   lambda_0=1,
                                                                   lat=lat_red,
                                                                   lon=lon_red,
                                                                   threshold=disp_threshold)
    # np.save(os.path.join(data_wd, fname), dispersal_probs)
    dispersal_probs_sparse = sparse.COO(dispersal_probs)
    sparse.save_npz(os.path.join(data_wd, fname_sparse), dispersal_probs_sparse)
# test
# rs = cn.get_rnd_gen(12345)
# lat_tmp = rs.integers(0, 5, (5, 5))
# lon_tmp = lat_tmp # np.random.randint(0, 5, (5, 5))
# precomputed_dispersal_probs = dispersalDistancesThresholdCoord(5,
#                                                           lambda_0=1,
#                                                           lat=lat_tmp,
#                                                           lon=lon_tmp,
#                                                           threshold=5)
#
# sns.heatmap(precomputed_dispersal_probs[0][0], cmap="YlGnBu")
# plt.title("Graph - SDM")
# plt.show()
# end test


# parse disturbance layers
disturbance_files = np.sort(glob.glob(env_data_wd + "/*.tif"))
layers = []
disturbance_names = []
for i in range(len(disturbance_files)):
    data, name = sdm_utils.get_sdm_data(disturbance_files, i, cropped=None)
    data /= np.nanmax(data)
    # data[reference_grid_pu == 0] = np.nan
    layers.append(data)
    disturbance_names.append(name)

disturbance_layers = np.squeeze(np.array(layers))
disturbance_layer_name = "Area_Swept_disturbance" # "Fishing"
f = disturbance_layers[disturbance_names.index(disturbance_layer_name)] + 0
f[f > 0] = np.log(f[f > 0])
f_rescaled = f + np.abs(np.nanmin(f))
f_rescaled /= np.nanmax(f_rescaled)
if min_disturbance:
    f_rescaled[f_rescaled < min_disturbance] = min_disturbance
disturbance_layers[disturbance_names.index(disturbance_layer_name)] = f_rescaled + 0
# log-transform fishing pressure

cost_layer_name = "Total_catch_cost_layer"
min_cost = 0.1
f = disturbance_layers[disturbance_names.index(disturbance_layer_name)] + 0
# f[f > 0] = np.log(f[f > 0])
f_rescaled = f + np.abs(np.nanmin(f))
f_rescaled[np.isnan(f_rescaled)] = np.abs(np.nanmin(f))
f_rescaled  = f_rescaled + min_cost
cost_layer = f_rescaled / np.nanmax(f_rescaled)

overall_disturbance = disturbance_layers[disturbance_names.index(disturbance_layer_name)]
selective_disturbance = disturbance_layers[disturbance_names.index(disturbance_layer_name)]
selective_disturbance *= max_disturbance

graph_cost = np.zeros(grid_length ** 2)
graph_cost[:n_pus] = cost_layer[reference_grid_pu > 0].flatten()
graph_cost = graph_cost.reshape((grid_length, grid_length))
graph_cost[np.isnan(graph_cost)] = 0


graph_disturbance = np.zeros(grid_length ** 2)
graph_disturbance[:n_pus] = overall_disturbance[reference_grid_pu > 0].flatten()
graph_disturbance = graph_disturbance.reshape((grid_length, grid_length))
graph_disturbance[np.isnan(graph_disturbance)] = 0

graph_selective_disturbance = np.zeros(grid_length ** 2)
graph_selective_disturbance[:n_pus] += selective_disturbance[reference_grid_pu > 0].flatten()
graph_selective_disturbance = graph_selective_disturbance.reshape((grid_length, grid_length))
graph_selective_disturbance[np.isnan(graph_selective_disturbance)] = 0

if do_plots:
    sns.heatmap(overall_disturbance, cmap="RdYlGn_r")
    plt.title("Disturbance")
    plt.show()

    sns.heatmap(selective_disturbance, cmap="RdYlGn_r")
    plt.title("Selective disturbance")
    plt.show()

    sns.heatmap(cost_layer)#, cmap="RdYlGn_r")
    plt.title("Cost")
    plt.show()

    for i in range(len(disturbance_files)):
        sns.heatmap(disturbance_layers[i] * reference_grid_pu.astype(int), cmap="RdYlGn_r")
        plt.title(disturbance_names[i])
        plt.show()



#--- env settings
r_seeds = [123, 124, 125, 126]
K_biome = np.ones((grid_length, grid_length))
max_K_cell = 100
max_K_multiplier = 10
rnd_sensitivity_seed = r_seeds[0]
size_protection_unit = np.array([1, 1]) # only 1x1 makes sense in graph-grid!
fixed_disturbance = True
edge_effect = 0
steps_fast_fw = 5
#--- env settings

# create n. individuals and carrying capacities
mask_suitability = (np.sum(graph_sdms, axis=0) > 0).astype(int)
# multiplies the disturbance to have 0 where 0 species are found

h3d, K_max, K_cells = cn_util.generate_h3d(graph_sdms, graph_suitability, mask_suitability,
                                           max_K_cell, K_biome, max_K_multiplier)

# placeholder for cost layer
cost_layer = np.ones((grid_length, grid_length))


# create SimGrid object
# initGrid with empirical 3d histogram
stateInitializer = cn.EmpiricalStateInit(h3d)

# initialize grid to reach carrying capacity
rs = cn.get_rnd_gen(rnd_sensitivity_seed)
disturbance_sensitivity = rs.random(n_species)

env2d = cn.BioDivEnv(budget=0.1,
                     gridInitializer=stateInitializer,
                     length=grid_length,
                     n_species=n_species,
                     K_max=K_max,
                     disturbance_sensitivity=disturbance_sensitivity,
                     disturbanceGenerator=cn.FixedEmpiricalDisturbanceGenerator(0),
                     # to fast forward to w 0 disturbance
                     dispersal_rate=5,
                     growth_rate=[2],
                     resolution=size_protection_unit,
                     habitat_suitability=graph_suitability,
                     cost_pu=cost_layer,
                     precomputed_dispersal_probs=dispersal_probs_sparse,
                     use_small_grid=True
                     )

# _ = env2d.step(skip_dispersal=False)

# test dispersal

# ep = np.einsum_path("sij,ijnm->snm", env2d.bioDivGrid.h, dispersal_probs_sparse.todense(), optimize=True)
# dd = dispersal_probs_sparse.todense()
# NumCandidates = np.einsum("sij,ijnm->snm", env2d.bioDivGrid.h, dispersal_probs_sparse)
#
# NumCandidates_l = np.array([sparse.einsum("ij,ijnm->nm",
#                                  env2d.bioDivGrid.h[i],
#                                  dispersal_probs_sparse).todense() for i in range(n_species)])
#
# res_td = sparse.tensordot(env2d.bioDivGrid.h, dispersal_probs_sparse)
#
#
# s = 20
# i = 240
# j = 240
# n = 240
# m = 240
# import numpy as np
# data3d = np.random.normal(size=s*i*j).reshape((s, i, j))
# data4d = np.random.normal(size=i*j*n*m).reshape((i, j, n, m))
#
# res = np.einsum('sij, ijnm -> snm', data3d, data4d, optimize=True)
# res_td = np.tensordot(data3d, data4d)
#
# np.allclose(res, res_td)
#
#
# task  ="""
# s = 3; i = 4; j = 5; n = 6; m = 7; import numpy as np; data3d = np.random.normal(size=s*i*j).reshape((s, i, j)); data4d = np.random.normal(size=i*j*n*m).reshape((i, j, n, m)); res = np.einsum('sij, ijnm -> snm', data3d, data4d, optimize=True);
# """
#
# import timeit
#
# print(timeit.timeit(stmt=task))
#

# end test



# evolve system to reach K_max
env2d.set_calc_reward(False)
env2d.fast_forward(steps_fast_fw, disturbance_effect_multiplier=0, verbose=True, skip_dispersal=False)

# cn.plot_env_state(env2d, wd=results_wd, species_list=[])



ext_risk_class = cn.ExtinctioRiskRedListEmpirical  #cn.ExtinctionRiskPopSize #

evolve_rl_status = True
use_empirical_rl_status = False
starting_rl_status=None
status = None # rs.integers(0, 5, env2d.bioDivGrid.numberOfSpecies())

ext_risk_obj = ext_risk_class(natural_state=env2d.grid_obj_previous,
                              current_state=env2d.bioDivGrid,
                              starting_rl_status=starting_rl_status,
                              evolve_status=evolve_rl_status,
                              relative_pop_thresholds=np.array([0.10,   # CR / EX: 0
                                                                0.30,   # EN: 1
                                                                0.50,   # VU: 2
                                                                0.60]), # NT: 3
                              epsilon=0.5,
                              # eps=1: last change, eps=0.5: rolling average, eps<0.5: longer legacy of long-term change
                              sufficient_protection=0.5,
                              pop_decrease_threshold=0.1)


variable_growth_rates = False
protection_fraction = 0.05

d = np.einsum('sxy->xy', h3d)
protection_steps = np.round(
    (d[d > 1].size * protection_fraction) / (size_protection_unit[0] * size_protection_unit[1])).astype(int)
print("protection_steps:", protection_steps)

if variable_growth_rates:
    rs = cn.get_rnd_gen(r_seeds[1])
    growth_rates = rs.beta(0.5, 0.5, env2d.bioDivGrid.numberOfSpecies())
    growth_rates /= np.mean(growth_rates)
else:
    growth_rates = [1]

mask_disturbance = (K_cells > 0).astype(int)
disturbanceGenerator = cn.FixedEmpiricalDisturbanceGenerator(0)
selective_disturbanceGenerator = cn.FixedEmpiricalDisturbanceGenerator(0)
init_disturbance = disturbanceGenerator.updateDisturbance(graph_disturbance * mask_disturbance)
init_selective_disturbance = selective_disturbanceGenerator.updateDisturbance(graph_selective_disturbance * mask_disturbance)
cost_layer = init_disturbance / np.max(init_disturbance)

outfile = "test"
out_tag = ""
degrade_steps = 5
budget = np.sum(graph_cost) * protection_fraction
models_wd = "/Users/dsilvestro/Software/CAPTAIN/captain-restore/trained_models"
model_file = "training_conserve_ext_risk.log"
reward = ["carbon", "ext_risk", "ext_risk_carbon"][1]
plot_sim = False
dynamic_print = True
actions_per_step = 100
feature_update_per_step = True # features are recomputed only if env.step() happens
heuristic_policies = [None, "random", "cheapest", "most_biodiverse", "most_natural_carbon", "highest_MSA", # 0:5
                          "most_natural_biodiversity", "most_current_carbon", "highest_STAR_t", "highest_STAR_r"] # 6:9
heuristic_policy = heuristic_policies[0]
minimize_policy = False
max_protection_level = 1

sensitivities = {
    'disturbance_sensitivity': rs.beta(0.5, 0.5, env2d.n_species),
    'selective_sensitivity': rs.beta(0.5, 0.5, env2d.n_species),
    'climate_sensitivity': rs.beta(0.5, 0.5, env2d.n_species)
}


# config simulation
config = cn.ConfigOptimPolicy(rnd_seed=r_seeds[2],
                              obsMode=1,
                              # 0: random, 1: full monitor, 2: citizen-science, 3: one-time, 4: value, 5: area
                              feature_update_per_step=feature_update_per_step,
                              steps=protection_steps,
                              simulations=1,
                              observePolicy=1,  # 0: NO-OBSERVE-UPDATE 1: ORACLE 2: PROTECTATONCE
                              disturbance=-1,
                              degrade_steps=degrade_steps,
                              initial_disturbance=init_disturbance,  # set initial disturbance matrix
                              edge_effect=edge_effect,
                              protection_cost=1,
                              n_nodes=[2, 2],
                              random_sim=0,
                              # "0: fixed (replicable) simulations; 1: random; 2: fixed training, seq pickle"
                              rewardMode=reward,
                              obs_error=0,  # "Amount of error in species counts (feature extraction)"
                              use_true_natural_state=True,
                              resolution=size_protection_unit,
                              grid_size=env2d.length,
                              budget=budget,
                              dispersal_rate=1,  # TODO: check if this can also be a vector per species
                              growth_rates=growth_rates,  # can be 1 values (list of 1 item) or or one value per species
                              use_climate=0,  # "0: no climate change, 1: climate change, 2: climate disturbance,
                              # 3: climate change + random variation"
                              rnd_alpha=0,
                              # (st.dev of sp.-specific fluctuation in mortality (if 'by_species' ==1 in SimpleGrid)
                              outfile=outfile + out_tag + ".log",
                              # model settings
                              trained_model=os.path.join(models_wd, model_file),
                              temperature=100,
                              deterministic_policy=1,  # 0: random policy (altered by temperature);
                              # 1: deterministic policy (overrides temperature)
                              sp_threshold_feature_extraction=1,
                              start_protecting=1,
                              plot_sim=plot_sim,
                              plot_species=[],
                              wd_output=results_wd,
                              grid_h=env2d.bioDivGrid.h,  # 3D hist of species (e.g. empirical)
                              distb_objects=[disturbanceGenerator, selective_disturbanceGenerator],
                              # list of distb_obj, selectivedistb_obj
                              return_env=True,
                              ext_risk_obj=ext_risk_obj,
                              # here set to 1 because env2d.bioDivGrid.h is already fast-evolved to carrying capcaity
                              max_K_multiplier=1,
                              suitability=graph_suitability,
                              cost_layer=cost_layer,
                              actions_per_step=actions_per_step,
                              heuristic_policy=heuristic_policy,
                              minimize_policy=minimize_policy,
                              use_empirical_setup=True,
                              max_protection_level=max_protection_level,
                              dynamic_print=dynamic_print,
                              precomputed_dispersal_probs=dispersal_probs_sparse,
                              use_small_grid=True,
                              sensitivities=sensitivities
                              )



# env_optim = cn.run_restore_policy(config)


# tests w/o protection
# config.budget = 1000
# config.steps = 100
env_optim = cn.run_restore_policy(config)

env_optim.getExtinction_risk_labels()

# check tot pop size
np.sum(env2d.bioDivGrid.h)
np.sum(env_optim.bioDivGrid.h)


env_optim.bioDivGrid.individualsPerSpecies() / env2d.bioDivGrid.individualsPerSpecies()

cn.plot_env_state(env_optim, wd=results_wd, species_list=[])

with open("", "wb") as output:  # Overwrites any existing file.
    cn.pickle.dump(env_optim, output, cn.pickle.HIGHEST_PROTOCOL)

# backtransform and plot
var = env_optim.bioDivGrid.protection_matrix
tmp = var.flatten()[:-(var.size - n_pus)] + 0
z = np.zeros(original_grid_shape)
z[reference_grid_pu > 0] += tmp
z[reference_grid_pu == 0] = np.nan
sns.heatmap(z, cmap="YlGn")
plt.title("Proposed areas for protection")
plt.show()














