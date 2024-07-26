import copy
import os, csv
import glob
import numpy as np
import copy
import pandas as pd

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
import argparse

SEED = 1234
policy_indx = 0

w_station = False

if w_station:
    p = argparse.ArgumentParser()
    p.add_argument('-p', type=int, help='heuristic policy indx', default=0)
    args = p.parse_args()
    policy_indx = args.p  # 0: carbon, 1: biodiv, 2: carbon & biodiv
    # 0: None, 1: random, 2: cheapest, 3: most biodiverse, 4: most_natural_carbon, 5: highest_MSA

data_wd = "/Users/dsilvestro/Documents/Projects/Ongoing/Captain_NZ/data"
if w_station: data_wd = "/data/captain_project/captain-restore/data"
env_data_wd = os.path.join(data_wd, "Disturbance and cost")
sdm_data_wd = os.path.join(data_wd, "Present SDMs")
result_dir = "res_"

sensitivity_alpha_beta = [0.5, 0.5]
max_disturbance = 0.99
min_disturbance = 0.72
max_selective_disturbance = 1
max_species_cuptoff = 20
if w_station:
    max_species_cuptoff = None
prob_threshold = 0.5 # truncate habitat suitability below
full_suitability = 0.9 # above this -> set to 1
do_plots = True
show_plots = False # if false save them as png
plot_env_states = False
if w_station:
    show_plots = False
    plot_env_states = True
min_cost = 0.1
disturbance_layer_name = "Area_Swept_disturbance" # "Fishing"
cost_layer_name = "Total_catch_cost_layer"
save_env_pkl = False

#--- env settings
r_seeds_dict = {'rnd_sensitivity': 123 + SEED,
                'rnd_growth': 124 + SEED,
                'rnd_config_policy': 125 + SEED,
                'rnd_k_species': 126 + SEED
                }
max_K_cell = 10000
max_K_multiplier = 10
use_K_species = True

size_protection_unit = np.array([1, 1]) # only 1x1 makes sense in graph-grid!
fixed_disturbance = True
zero_disturbance = False
edge_effect = 0
steps_fast_fw = 0
#---

#--- MISC settings
variable_growth_rates = False
protection_fraction = 0.05
outfile = "res_" + str(SEED) + "_"
out_tag = ""
degrade_steps = 5
models_wd = "/Users/dsilvestro/Software/CAPTAIN/captain-restore/trained_models"
if w_station: models_wd = "/data/captain_project/captain-restore/trained_models"
model_file = "empirical_training.log" #"training_ext_risk.log" #"training_20_ext_risk.log"
reward = ["carbon", "ext_risk", "ext_risk_carbon", "ext_risk_protect"][3]
plot_sim = False
dynamic_print = True
actions_per_step = 100
feature_update_per_step = True # features are recomputed only if env.step() happens
heuristic_policies = [None, "random", "cheapest", "most_biodiverse", "most_natural_carbon", "highest_MSA", # 0:5
                          "most_natural_biodiversity", "most_current_carbon", "highest_STAR_t", "highest_STAR_r", # 6:9
                      "no_protection"] # 10
heuristic_policy = heuristic_policies[policy_indx]
minimize_policy = False
max_protection_level = 1
budget = None # if budget = None: np.sum(costs) * protection_fraction
#---

#--- Extinction risk settings
ext_risk_class = cn.ExtinctioRiskCompareNatural #cn.ExtinctioRiskRedListEmpirical  #cn.ExtinctionRiskPopSize #
evolve_rl_status = True
starting_rl_status = None
rnd_starting_rl_status = False
empirically_threatened = np.sort(np.array([35, 165, 174, 181, 195, 201, 203, 243, 273, 330, 347, 445, 1, 9, 98, 122, 177, 216, 260, 277, 297, 376, 387, 413, 357]) - 1)
empirically_declining = np.sort(np.array([35, 165, 174, 181, 195, 201, 203, 243, 273, 330, 347, 445]) -1)
pop_decrease_threshold = 0.01
sufficient_protection = 0.5
min_individuals_cell = 0
relative_pop_thresholds = np.array([0.10,  # CR / EX: 0
                                    0.30,  # EN: 1
                                    0.50,  # VU: 2
                                    0.60])  # NT: 3
""" previously:
0.35,   # CR / EX: 0
0.50,   # EN: 1
0.70,   # VU: 2
0.80]),"""
#---

# create out
if heuristic_policy == "no_protection":
    heuristic_policy = "random"
    policy_indx = 1
    budget = 0

if heuristic_policy is None:
    results_wd = os.path.join(data_wd, result_dir + str(SEED))
else:
    results_wd = os.path.join(data_wd, result_dir + heuristic_policy + "_" + str(SEED))
try:
    os.mkdir(results_wd)
except FileExistsError:
    pass

# parse SDM files and create 3D object
sdms, species_names = sdm_utils.get_data_from_list(sdm_data_wd, tag="/*.tif", max_species_cuptoff=max_species_cuptoff)

sdms[sdms > full_suitability] = 1
suitability = cn_util.get_habitat_suitability(sdms, prob_threshold)
species_richness = np.nansum(suitability, axis=0)
original_grid_shape = species_richness.shape
max_species_richness = np.nanmax(species_richness)

# MAKE IT A GRAPH without gaps
reference_grid_pu = species_richness > 0
reference_grid_pu_nan = reference_grid_pu.astype(float)
reference_grid_pu_nan[reference_grid_pu_nan == 0] = np.nan
n_species = sdms.shape[0]


if do_plots:
    cn_util.plot_map(species_richness, z=reference_grid_pu_nan, nan_to_zero=False, vmax=max_species_richness,
                     cmap="YlGnBu", show=show_plots, title="Species richness (Natural state)",
                     outfile=os.path.join(results_wd, "species_richness_natural.png"), dpi=250)
    cn_util.plot_map(reference_grid_pu, z=reference_grid_pu_nan, nan_to_zero=True,
                     show=show_plots, title="Reference grid",
                     outfile=os.path.join(results_wd, "reference_grid_pu.png"), dpi=250)

# graph SDMs
graph_sdms, n_pus, grid_length = sdm_utils.grid_to_graph(sdms, reference_grid_pu)
graph_suitability = cn_util.get_habitat_suitability(graph_sdms, prob_threshold,integer=True)

# reduce coordinates
xy_coords = np.meshgrid(np.arange(original_grid_shape[1]), np.arange(original_grid_shape[0]))
graph_coords, _, __ = sdm_utils.grid_to_graph(np.array(xy_coords), reference_grid_pu)

# dispersal threshold (in number of cells)
disp_threshold = 3
fname_sparse = "disp_probs_sp%s_th%s.npz" % (n_species, disp_threshold)
try:
    dispersal_probs_sparse = sparse.load_npz(os.path.join(data_wd, fname_sparse))
except:
    dispersal_probs = cn_sim_grid.dispersalDistancesThresholdCoord(grid_length,
                                                                   lambda_0=1,
                                                                   lat=graph_coords[1],
                                                                   lon=graph_coords[0],
                                                                   threshold=disp_threshold)
    # np.save(os.path.join(data_wd, fname), dispersal_probs)
    dispersal_probs_sparse = sparse.COO(dispersal_probs)
    sparse.save_npz(os.path.join(data_wd, fname_sparse), dispersal_probs_sparse)

# parse disturbance layers
disturbance_files = np.sort(glob.glob(env_data_wd + "/*.tif"))

disturbance_layers, disturbance_names = sdm_utils.get_data_from_list(env_data_wd, tag="/*.tif", rescale=True)

# set and rescale disturbance layer
f = disturbance_layers[disturbance_names.index(disturbance_layer_name)] + 0
f[f > 0] = np.log(f[f > 0]) # log-transform fishing pressure
f_rescaled = f + np.abs(np.nanmin(f))
f_rescaled /= np.nanmax(f_rescaled)
if min_disturbance:
    f_rescaled[np.isnan(f_rescaled)] = 0
    f_rescaled[np.isnan(reference_grid_pu_nan)] = np.nan
    f_rescaled[f_rescaled < min_disturbance] = min_disturbance
disturbance_layers[disturbance_names.index(disturbance_layer_name)] = f_rescaled + 0
overall_disturbance = disturbance_layers[disturbance_names.index(disturbance_layer_name)] * max_disturbance
selective_disturbance = disturbance_layers[disturbance_names.index(disturbance_layer_name)] * max_selective_disturbance

# set and rescale cost layer
f = disturbance_layers[disturbance_names.index(disturbance_layer_name)] + 0
f_rescaled = f + np.abs(np.nanmin(f))
f_rescaled[np.isnan(f_rescaled)] = np.abs(np.nanmin(f))
f_rescaled  = f_rescaled + min_cost
cost_layer = f_rescaled / np.nanmax(f_rescaled)

# graphs of cost and disturbance
graph_disturbance, _, __ = sdm_utils.grid_to_graph(overall_disturbance, reference_grid_pu, n_pus, nan_to_zero=True)
graph_selective_disturbance, _, __ = sdm_utils.grid_to_graph(selective_disturbance, reference_grid_pu, n_pus, nan_to_zero=True)
graph_cost, _, __ = sdm_utils.grid_to_graph(cost_layer, reference_grid_pu, n_pus, nan_to_zero=True)


if do_plots:
    cn_util.plot_map(overall_disturbance, z=reference_grid_pu_nan, nan_to_zero=False,
                     cmap="RdYlBu_r", show=show_plots, title="Disturbance", vmin=0,
                     outfile=os.path.join(results_wd, "disturbance.png"), dpi=250)

    cn_util.plot_map(selective_disturbance, z=reference_grid_pu_nan, nan_to_zero=True,
                     cmap="RdYlBu_r", show=show_plots, title="Selective disturbance", vmin=0,
                     outfile=os.path.join(results_wd, "selective_disturbance.png"), dpi=250)

    cn_util.plot_map(cost_layer, z=reference_grid_pu_nan, nan_to_zero=True,
                     cmap="autumn_r", show=show_plots, title="Costs",
                     outfile=os.path.join(results_wd, "costs.png"), dpi=250, vmin=0)


K_biome = np.ones((grid_length, grid_length)) # * np.sum(graph_sdms, axis=0) <- K dependent of species richness
# create n. individuals and carrying capacities
mask_suitability = (np.sum(graph_sdms, axis=0) > 0).astype(int)
# mask_suitability multiplies the disturbance that is therefore set to 0 where 0 species are found

h3d, K_max, K_cells = cn_util.generate_h3d(graph_sdms, graph_suitability, mask_suitability,
                                           max_K_cell, K_biome, max_K_multiplier)

if use_K_species:
    # species-specific carrying capacities
    rs_k = cn.get_rnd_gen(r_seeds_dict['rnd_k_species'])
    K_species = 2 + rs_k.random(n_species) * 10
    # make empirically rare species rare
    emp_threat = empirically_threatened[empirically_threatened < n_species]
    K_species[emp_threat] = np.sort(K_species)[:len(emp_threat)]
    K_species_3D = K_species[:, np.newaxis, np.newaxis] * graph_sdms * graph_suitability #np.ones(h3d.shape)
    h3d = K_species_3D + 0
else:
    K_species = None
    K_species_3D = None

# create SimGrid object: initGrid with empirical 3d histogram
stateInitializer = cn.EmpiricalStateInit(h3d)



# initialize grid to reach carrying capacity: no disturbance, no sensitivity
env2d = cn.BioDivEnv(budget=0.1,
                     gridInitializer=stateInitializer,
                     length=grid_length,
                     n_species=n_species,
                     K_max=K_max,
                     disturbance_sensitivity=np.zeros(n_species),
                     disturbanceGenerator=cn.FixedEmpiricalDisturbanceGenerator(0),
                     # to fast forward to w 0 disturbance
                     dispersal_rate=5,
                     growth_rate=[2],
                     resolution=size_protection_unit,
                     habitat_suitability=graph_suitability,
                     cost_pu=graph_cost,
                     precomputed_dispersal_probs=dispersal_probs_sparse,
                     use_small_grid=True,
                     K_species=K_species_3D
                     )



# evolve system to reach K_max
env2d.set_calc_reward(False)
env2d.fast_forward(steps_fast_fw, disturbance_effect_multiplier=0, verbose=False, skip_dispersal=False)

if do_plots:
    species_richness_init =  sdm_utils.graph_to_grid(env2d.bioDivGrid.speciesPerCell(), reference_grid_pu)
    cn_util.plot_map(species_richness_init, z=reference_grid_pu_nan, nan_to_zero=False, vmax=max_species_richness,
                     cmap="YlGnBu", show=show_plots, title="Species richness (Natural state - FWD)",
                     outfile=os.path.join(results_wd, "species_richness_natural_FWD.png"), dpi=250)

    pop_density =  sdm_utils.graph_to_grid(env2d.bioDivGrid.individualsPerCell(), reference_grid_pu)
    cn_util.plot_map(pop_density, z=reference_grid_pu_nan, nan_to_zero=False,
                     cmap="Greens", show=show_plots, title="Population density (Natural state - FWD)",
                     outfile=os.path.join(results_wd, "population_density_natural_FWD.png"), dpi=250)

    # sns.barplot(x=np.arange(n_species), y=env2d.bioDivGrid.individualsPerSpecies())
    # plt.show()


if plot_env_states:
    cn.plot_env_state(env2d, wd=results_wd, species_list=[])

# set species sensitivities
rs_s = cn.get_rnd_gen(r_seeds_dict['rnd_sensitivity'])
sensitivities = {
    'disturbance_sensitivity': rs_s.beta(sensitivity_alpha_beta[0], sensitivity_alpha_beta[1], n_species),
    'selective_sensitivity': rs_s.beta(sensitivity_alpha_beta[0], sensitivity_alpha_beta[1], n_species) * 0,
    'climate_sensitivity': rs_s.beta(sensitivity_alpha_beta[0], sensitivity_alpha_beta[1], n_species)
}


# make empirically rare / threatened species sensitive
emp_declining = empirically_declining[empirically_declining < n_species]
sensitivities['disturbance_sensitivity'][emp_declining] = np.sort(sensitivities['disturbance_sensitivity'])[::-1][:len(emp_declining)]
sensitivities['selective_sensitivity'][emp_declining] = np.sort(sensitivities['selective_sensitivity'])[::-1][:len(emp_declining)]


# set extinction risks
if rnd_starting_rl_status:
    starting_rl_status = np.random.randint(0, 4, n_species)

ext_risk_obj = ext_risk_class(natural_state=env2d.grid_obj_previous,
                              current_state=env2d.bioDivGrid,
                              starting_rl_status=starting_rl_status,
                              evolve_status=evolve_rl_status,
                              relative_pop_thresholds=relative_pop_thresholds,
                              epsilon=0.5,
                              # eps=1: last change, eps=0.5: rolling average, eps<0.5: longer legacy of long-term change
                              sufficient_protection=0.5,
                              pop_decrease_threshold=pop_decrease_threshold,
                              min_individuals_cell=min_individuals_cell)


d = np.einsum('sxy->xy', h3d)
protection_steps = np.round(
    (d[d > 1].size * protection_fraction) / (size_protection_unit[0] * size_protection_unit[1])).astype(int)
print("protection_steps:", protection_steps)

if variable_growth_rates:
    rs_g = cn.get_rnd_gen(r_seeds_dict['rnd_growth'])
    growth_rates = rs_g.beta(0.5, 0.5, n_species)
    growth_rates /= np.mean(growth_rates)
else:
    growth_rates = 1

mask_disturbance = (K_cells > 0).astype(int)
disturbanceGenerator = cn.FixedEmpiricalDisturbanceGenerator(0)
selective_disturbanceGenerator = cn.FixedEmpiricalDisturbanceGenerator(0)
if zero_disturbance:
    zd = 0
else: zd = 1
init_disturbance = disturbanceGenerator.updateDisturbance(graph_disturbance * mask_disturbance * zd)
init_selective_disturbance = selective_disturbanceGenerator.updateDisturbance(graph_selective_disturbance * mask_disturbance * zd)

if budget is None:
    # budget set as a function of costs and target protection fraction
    budget = np.sum(graph_cost) * protection_fraction

# config simulation
config = cn.ConfigOptimPolicy(rnd_seed=r_seeds_dict['rnd_config_policy'],
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
                              n_nodes=[1, 0], #[2, 2],
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
                              sp_threshold_feature_extraction=0.001,
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
                              sensitivities=sensitivities,
                              K_species_3D=K_species_3D,
                              pre_steps=10
                              )

config_init = copy.deepcopy(config)
config_init.steps = 5
config_init.budget = 0
config_init.actions_per_step = 1
env_init = cn.run_restore_policy(config_init)

#####
env = copy.deepcopy(env_init)
env.iterations = 1002
env.budget = np.sum(graph_cost) * protection_fraction
actions_per_step = 100
env.reset_init_values()
env.set_calc_reward(True)
env_optim = cn.run_policy_empirical_env(config, env)



######






simulated_init_status = env_init.getExtinction_risk_labels()
# simulated_init_status[emp_threat]

print(env_init.species_risk_criteria.population_change)
print(env_init.bioDivGrid.individualsPerSpecies())

current_sp_range = env_init.bioDivGrid.geoRangePerSpecies()
current_pop_sizes = env_init.bioDivGrid.individualsPerSpecies()
current_protected_pop_sizes = env_init.bioDivGrid.protectedIndPerSpecies()
current_protected_range = env_init.bioDivGrid.protectedRangePerSpecies()

x= np.array([current_protected_range, env_init.species_risk_criteria.population_change]).T


population_change_to_init = current_pop_sizes / env_init.species_risk_criteria._init_pop_size - 1
population_change_to_natural = current_pop_sizes / env_init.species_risk_criteria._natural_pop_size



np.digitize(population_change_to_natural, bins=env_init.species_risk_criteria.relative_pop_thresholds)

#  n / K ratio
i =2
(1 - env_init.bioDivGrid.h[i] / np.maximum(1, env_init.bioDivGrid._K_species[i] )) * graph_suitability[i]

print(
    np.array([
        env_init.grid_obj_previous.individualsPerSpecies(),
        env_init.bioDivGrid.individualsPerSpecies(),
        env_init.species_risk_criteria._init_pop_size,
        env_init.species_risk_criteria._natural_pop_size,
        np.einsum('sxy -> s', K_species_3D),
        env_init.getExtinction_risk_labels(),
        sensitivities["disturbance_sensitivity"]
    ]).T
)

current_pop_sizes / env_init.species_risk_criteria._natural_pop_size -1

env_init.species_risk_criteria.classify_species(env_init.bioDivGrid)


i = 0
pop_density =  sdm_utils.graph_to_grid(env_init.bioDivGrid.individualsPerCell(), reference_grid_pu)
cn_util.plot_map(pop_density, z=reference_grid_pu_nan, nan_to_zero=False,
                 cmap="Greens", show=True, title="Population density (protection)",
                 outfile=os.path.join(results_wd, "population_density_after_protection.png"), dpi=250)

cn_util.plot_map(sdms[i], z=reference_grid_pu_nan, nan_to_zero=False,
                 show=True, title="Species %s (SDM)" % i)

species_richness_init = sdm_utils.graph_to_grid(env_init.bioDivGrid.h[i], reference_grid_pu)
cn_util.plot_map(np.log(1 + species_richness_init), z=reference_grid_pu_nan, nan_to_zero=False,
                 show=True, title="Species %s (current)" % i)

species_richness_init = sdm_utils.graph_to_grid(env2d.bioDivGrid.h[i], reference_grid_pu)
cn_util.plot_map(np.log(1 + species_richness_init), z=reference_grid_pu_nan, nan_to_zero=False,
                 show=True, title="Species %s (init)" % i)

species_richness_init = sdm_utils.graph_to_grid(K_species_3D[i], reference_grid_pu)
cn_util.plot_map(np.log(1 + species_richness_init), z=reference_grid_pu_nan, nan_to_zero=False,
                 show=True, title="Species %s (h3d)" % i)



#
# plt.scatter(env_init.species_risk_criteria.population_change, sensitivities['disturbance_sensitivity'])
# plt.show()
#
# plt.hist(sensitivities['disturbance_sensitivity'])
# plt.show()
#
# # plot CR species
#
# cr_indx = np.where(simulated_init_status == 0)[0]
#
# for i in cr_indx:
#     stats = np.array([sensitivities['disturbance_sensitivity'][i], sensitivities['selective_sensitivity'][i],
#           K_species[i], np.nansum(sdms[i]), simulated_init_status[i]])
#     print(stats)
#     cn_util.plot_map(sdms[i], z=reference_grid_pu_nan, nan_to_zero=False,
#                      show=True, title="Species %s" % i)
#



#
# species_richness_init = sdm_utils.graph_to_grid(env_init.bioDivGrid.h[cr_indx[i]], reference_grid_pu)
# cn_util.plot_map(species_richness_init, z=reference_grid_pu_nan, nan_to_zero=False,
#                  show=True, title="Species %s" % i)
#
#
#
#
#
# for i in empirically_threatened:
#     stats = np.array([sensitivities['disturbance_sensitivity'][i], sensitivities['selective_sensitivity'][i],
#           K_species[i], np.nansum(sdms[i]), simulated_init_status[i]])
#     print(stats)
#
#
#
# simulated_init_status[empirically_threatened[3]]
# cn_util.plot_map(sdms[empirically_threatened[3]], z=reference_grid_pu_nan, nan_to_zero=False,
#                  show=True, title="Species %s" % i)
#
#




if plot_env_states:
    cn.plot_env_state(env_init, wd=results_wd, species_list=[])

if do_plots:
    species_richness_init =  sdm_utils.graph_to_grid(env_init.bioDivGrid.speciesPerCell(), reference_grid_pu)
    cn_util.plot_map(species_richness_init, z=reference_grid_pu_nan, nan_to_zero=False, vmax=max_species_richness,
                     cmap="YlGnBu", show=show_plots, title="Species richness (Disturbed)",
                     outfile=os.path.join(results_wd, "species_richness_disturbed.png"), dpi=250)

    pop_density =  sdm_utils.graph_to_grid(env_init.bioDivGrid.individualsPerCell(), reference_grid_pu)
    cn_util.plot_map(pop_density, z=reference_grid_pu_nan, nan_to_zero=False,
                     cmap="Greens", show=show_plots, title="Population density (Disturbed)",
                     outfile=os.path.join(results_wd, "population_density_disturbed.png"), dpi=250)


    # sns.barplot(x=np.arange(n_species), y=env_init.bioDivGrid.individualsPerSpecies(1), color="b")
    # plt.show()
    #
    # sns.barplot(x=np.arange(n_species), y=env2d.bioDivGrid.geoRangePerSpecies())
    # plt.show()


# save range table
# sp_ranges = env2d.bioDivGrid.geoRangePerSpecies()
# tbl = pd.DataFrame(np.array([species_names, sp_ranges]).T)
# tbl.columns = ["Species", "Range"]
# tbl.to_csv(os.path.join(data_wd, "species_table.csv"))
"""

(env_init.bioDivGrid.individualsPerSpecies(1) / env2d.bioDivGrid.individualsPerSpecies(1))

list(zip(species_names, simulated_init_status))

np.einsum('sxy -> s', graph_suitability)
env_init.bioDivGrid.geoRangePerSpecies()
env_init.bioDivGrid.geoRangePerSpecies() / env2d.bioDivGrid.geoRangePerSpecies()

"""
# config.growth_rates = config.growth_rates * 100
# config.steps = 1000
# env_optim = cn.run_restore_policy(config)


####
protected_range_fraction = env_optim.bioDivGrid.protectedRangePerSpecies() / env_optim.bioDivGrid.geoRangePerSpecies(0.1)

protected_range_fraction
env_optim.getExtinction_risk_labels()

print(
    np.array([
        env_optim.grid_obj_previous.individualsPerSpecies(),
        env_optim.bioDivGrid.individualsPerSpecies(),
        env_optim.species_risk_criteria._init_pop_size,
        env_optim.species_risk_criteria._natural_pop_size,
        # np.einsum('sxy -> s', K_species_3D),
        # env_init.getExtinction_risk_labels(),
        env_optim.getExtinction_risk_labels(),
        protected_range_fraction,
        # sensitivities["disturbance_sensitivity"]
    ]).T
)





####g
env_optim.getExtinction_risk_labels()

cn.print_update("\nSaving results and optimized env...")

if save_env_pkl:
    with open(os.path.join(results_wd, "env_optim.pkl"), "wb") as output:  # Overwrites any existing file.
        cn.pickle.dump(env_optim, output, cn.pickle.HIGHEST_PROTOCOL)

if plot_env_states:
    cn.plot_env_state(env_optim, wd=results_wd, species_list=[])


species_richness_init =  sdm_utils.graph_to_grid(env_optim.bioDivGrid.speciesPerCell(), reference_grid_pu)
pop_density =  sdm_utils.graph_to_grid(env_optim.bioDivGrid.individualsPerCell(), reference_grid_pu)
protection_matrix = sdm_utils.graph_to_grid(env_optim.bioDivGrid.protection_matrix, reference_grid_pu)

if do_plots:
    cn_util.plot_map(species_richness_init, z=reference_grid_pu_nan, nan_to_zero=False, vmax=max_species_richness,
                     cmap="YlGnBu", show=show_plots, title="Species richness (protection)",
                     outfile=os.path.join(results_wd, "species_richness_after_protection.png"), dpi=250)

    cn_util.plot_map(pop_density, z=reference_grid_pu_nan, nan_to_zero=False,
                     cmap="Greens", show=show_plots, title="Population density (protection)",
                     outfile=os.path.join(results_wd, "population_density_after_protection.png"), dpi=250)

    cn_util.plot_map(protection_matrix, z=reference_grid_pu_nan, nan_to_zero=False,
                     cmap="YlGn", show=show_plots, title="Proposed areas for protection",
                     outfile=os.path.join(results_wd, "optim_protected_areas.png"), dpi=250)







########
# calc final metrics
m = cn.calc_MSA(env_optim.grid_obj_previous.h, env_optim.bioDivGrid.h)
p = cn.calc_PDF_from_grid(env_optim.grid_obj_previous.individualsPerSpecies(),
                          env_optim.bioDivGrid.individualsPerSpecies())

st, sr = cn.calc_STAR_from_grid(env_optim.grid_obj_previous.h, env_optim.bioDivGrid.h,
                                sp_natural_range=env_optim.grid_obj_previous.geoRangePerSpecies(),
                                sp_ext_risk=env_optim.getExtinction_risk_labels(),
                                quandrant_grid_indx=env_optim._quandrant_grid_indx)

t = np.mean(st[st > 0])
r = np.mean(sr[sr > 0])

print("""
 MSA: %s
 PDF: %s
 STAR-t: %s
 STAR-r: %s

 """ % (m, p, t, r)
      )

print("Init risk_label_counts:", env_init.risk_label_counts())
print("Final risk_label_counts:", env_optim.risk_label_counts())


protected_range_fraction = env_optim.bioDivGrid.protectedRangePerSpecies() / env_optim.bioDivGrid.geoRangePerSpecies(0.1)
# average fraction of protected range per threat class (initial classification)
protected_fr_risk_class = cn.scipy.ndimage.mean(protected_range_fraction,
                                                labels=env_optim._init_sp_ext_risk,
                                                index=np.arange(5))

protected_ranges = [np.mean(protected_range_fraction),
                    np.min(protected_range_fraction),
                    np.max(protected_range_fraction)] + list(protected_fr_risk_class)


summary_file = os.path.join(results_wd, outfile + "_summary_stats.txt")
sim_i = 0
with open(summary_file, "w") as f:
    writer = csv.writer(f, delimiter="\t")
    head = ["Seed", "Disturbance", "RestoredCells",
            "CostRestoredCells", "Budget_left",
            "Carbon", "TotPopSize",
            "ExtantSpecies", "ExtinctSpecies",
            "MSA", "PDF", "STAR-t", "STAR-r",
            "CR", "EN", "VU", "NT", "LC",
            "Mean_pr", "Min_pr", "Max_pr",
            "CR_pr", "EN_pr", "VU_pr", "NT_pr", "LC_pr"]
    writer.writerow(head)

with open(summary_file, "a") as f:
    writer = csv.writer(f, delimiter="\t")
    head = [SEED,
            np.mean(env_optim.bioDivGrid.disturbance_matrix[env_optim.bioDivGrid.disturbance_matrix > 0]),
            env_optim.bioDivGrid.protection_matrix[env_optim.bioDivGrid.protection_matrix > 0].size,
            env_optim.cost_protected_quadrants, env_optim.budget,
            np.log10(env_optim.current_carbon), np.log10(np.sum(env_optim.bioDivGrid.h)),
            env_optim.bioDivGrid.numberOfSpecies(),
            env2d.bioDivGrid.numberOfSpecies() - env_optim.bioDivGrid.numberOfSpecies(),
            m, p, t, r] + list(env_optim.risk_label_counts()) + protected_ranges

    writer.writerow(head)

np.savez(os.path.join(results_wd, outfile + ""), )

current_sp_range = env_optim.bioDivGrid.geoRangePerSpecies()
current_pop_sizes = env_optim.bioDivGrid.individualsPerSpecies(min_individuals_cell)
current_protected_pop_sizes = env_optim.bioDivGrid.protectedIndPerSpecies(min_individuals_cell)
current_protected_range = env_optim.bioDivGrid.protectedRangePerSpecies()

x= np.array([current_protected_range, env_optim.species_risk_criteria.population_change]).T


population_change_to_init = current_pop_sizes / env_optim.species_risk_criteria._init_pop_size - 1


np.savez_compressed(
    file=os.path.join(results_wd, outfile + "stats.npz"),
    protection_matrix=protection_matrix,
    protected_range_fraction=protected_range_fraction,
    simulated_init_status=simulated_init_status,
    resulting_status=env_optim.getExtinction_risk_labels(),
    history=np.array(env_optim.history),
    history_var_names=env_optim.history_var_names,
    current_sp_range=current_sp_range,
    current_pop_sizes=current_pop_sizes,
    current_protected_pop_sizes=current_protected_pop_sizes,
    current_protected_range=current_protected_range

)


#
#
# # load results
# res_dir = "/Users/dsilvestro/Documents/Projects/Ongoing/Captain_NZ/test_res"
#           # "/Users/dsilvestro/Documents/Projects/Ongoing/Captain_NZ/test_res/env_optim.pkl"
# res_file = "env_optim.pkl"
# env_optim = cn.load_pickle_file(os.path.join(res_dir, res_file))
#
# # env_optim = cn.load_pickle_file()
#
# import pickle
#
# with open(pkl, "rb") as pkl:
#     list_cells = pickle.load(pkl)
#
#

# ## test
# l = cn.get_sp_indx_per_quadrant(env_optim.bioDivGrid.h, env_optim._quandrant_grid_indx)
#
# from scipy import ndimage
# max_cell_individual_sp_quadrant = ndimage.maximum(env_optim.bioDivGrid.h,
#                                                   labels= env_optim._quandrant_grid_indx[2],
#                                                   index= env_optim._quandrant_grid_indx[3])
#
#
#
# sp_id, quadrant_id = np.where(max_cell_individual_sp_quadrant >= 0.1)
#
#
#
# # l = [list(sp_id[quadrant_id == i]) for i in np.unique(quadrant_id)]
# l2 = [list(sp_id[quadrant_id == i]) for i in range(env_optim._quandrant_grid_indx[3].shape[1])]
# # print("[sp_id[quadrant_id == i] for i in np.unique(quadrant_id)]", len(l))
#
# l = []
# for i in range(env_optim._quandrant_grid_indx[3].shape[1]):
#     l.append( list(np.where(max_cell_individual_sp_quadrant[:,i] >= 0.1)[0]) )
#
#
#






