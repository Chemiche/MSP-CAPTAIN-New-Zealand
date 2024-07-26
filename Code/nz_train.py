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

w_station = False
SEED = 1234
BATCH_SIZE = 6
STEPS = 1002

policy_indx = 0

data_wd = "/Users/dsilvestro/Documents/Projects/Ongoing/Captain_NZ/data"
models_wd = "/Users/dsilvestro/Software/CAPTAIN/captain-restore/trained_models"
if w_station:
    models_wd = "/data/captain_project/captain-restore/trained_models"
    data_wd = "/data/captain_project/captain-restore/data"

env_data_wd = os.path.join(data_wd, "Disturbance and cost")
sdm_data_wd = os.path.join(data_wd, "Present SDMs")
result_dir = "train_emp_"

sensitivity_alpha_beta = [0.5, 0.5]
max_disturbance = 0.99
min_disturbance = 0.3
max_selective_disturbance = 1
if w_station:
    max_species_cuptoff = None
else:
    max_species_cuptoff = 20
    BATCH_SIZE = 1

prob_threshold = 0.5  # truncate habitat suitability below
full_suitability = 0.9  # above this -> set to 1
do_plots = False
show_plots = False  # if false save them as png
plot_env_states = False

min_cost = 0.1
disturbance_layer_name = "Area_Swept_disturbance"  # "Fishing"
cost_layer_name = "Total_catch_cost_layer"
save_env_pkl = False

# --- env settings

max_K_cell = 10000
max_K_multiplier = 10
use_K_species = True

size_protection_unit = np.array([1, 1])  # only 1x1 makes sense in graph-grid!
fixed_disturbance = True
zero_disturbance = False
edge_effect = 0
steps_fast_fw = 0
# ---

# --- MISC settings
variable_growth_rates = False
protection_fraction = 0.02
outfile = "train_emp_" + str(SEED) + "_"
out_tag = ""
degrade_steps = 5
model_file = "training_ext_risk.log"  # "training_20_ext_risk.log"
reward_indx = 5
reward = ["carbon", "ext_risk", "ext_risk_carbon", "ext_risk_protect", "star_t", "sp_risk_protect"][reward_indx]
plot_sim = False
dynamic_print = True
actions_per_step = 100
feature_update_per_step = True  # features are recomputed only if env.step() happens
heuristic_policies = [None, "random", "cheapest", "most_biodiverse", "most_natural_carbon", "highest_MSA",  # 0:5
                      "most_natural_biodiversity", "most_current_carbon", "highest_STAR_t", "highest_STAR_r",  # 6:9
                      "no_protection"]  # 10
heuristic_policy = None
minimize_policy = False
max_protection_level = 1
budget = None  # if budget = None: np.sum(costs) * protection_fraction
# ---

# --- Extinction risk settings
ext_risk_class = cn.ExtinctioRiskCompareNatural  # cn.ExtinctioRiskRedListEmpirical  #cn.ExtinctionRiskPopSize #
evolve_rl_status = True
starting_rl_status = None
empirically_threatened = np.sort(np.array(
    [35, 165, 174, 181, 195, 201, 203, 243, 273, 330, 347, 445, 1, 9, 98, 122, 177, 216, 260, 277, 297, 376, 387, 413,
     357]) - 1)
empirically_declining = np.sort(np.array([35, 165, 174, 181, 195, 201, 203, 243, 273, 330, 347, 445]) - 1)
pop_decrease_threshold = 0.01
sufficient_protection = 0.5
min_individuals_cell = 1
relative_pop_thresholds = np.array([0.10,  # CR / EX: 0
                                    0.30,  # EN: 1
                                    0.50,  # VU: 2
                                    0.60])  # NT: 3
""" previously:
0.35,   # CR / EX: 0
0.50,   # EN: 1
0.70,   # VU: 2
0.80]),"""
# ---

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
sdms, species_names = sdm_utils.get_data_from_list(sdm_data_wd, tag="/*.tif",
                                                   max_species_cuptoff=max_species_cuptoff)

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
graph_suitability = cn_util.get_habitat_suitability(graph_sdms, prob_threshold, integer=True)

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
f[f > 0] = np.log(f[f > 0])  # log-transform fishing pressure
f_rescaled = f + np.abs(np.nanmin(f))
f_rescaled /= np.nanmax(f_rescaled)
if min_disturbance:
    f_rescaled[np.isnan(f_rescaled)] = 0
    f_rescaled[np.isnan(reference_grid_pu_nan)] = np.nan
    f_rescaled[f_rescaled < min_disturbance] = min_disturbance
disturbance_layers[disturbance_names.index(disturbance_layer_name)] = f_rescaled + 0
overall_disturbance = disturbance_layers[disturbance_names.index(disturbance_layer_name)] * max_disturbance
selective_disturbance = disturbance_layers[
                            disturbance_names.index(disturbance_layer_name)] * max_selective_disturbance

# set and rescale cost layer
f = disturbance_layers[disturbance_names.index(disturbance_layer_name)] + 0
f_rescaled = f + np.abs(np.nanmin(f))
f_rescaled[np.isnan(f_rescaled)] = np.abs(np.nanmin(f))
f_rescaled = f_rescaled + min_cost
cost_layer = f_rescaled / np.nanmax(f_rescaled)

# graphs of cost and disturbance
graph_disturbance, _, __ = sdm_utils.grid_to_graph(overall_disturbance, reference_grid_pu, n_pus, nan_to_zero=True)
graph_selective_disturbance, _, __ = sdm_utils.grid_to_graph(selective_disturbance, reference_grid_pu, n_pus,
                                                             nan_to_zero=True)
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

K_biome = np.ones((grid_length, grid_length))  # * np.sum(graph_sdms, axis=0) <- K dependent of species richness
# create n. individuals and carrying capacities
mask_suitability = (np.sum(graph_sdms, axis=0) > 0).astype(int)
# mask_suitability multiplies the disturbance that is therefore set to 0 where 0 species are found

h3d, K_max, K_cells = cn_util.generate_h3d(graph_sdms, graph_suitability, mask_suitability,
                                           max_K_cell, K_biome, max_K_multiplier)

if budget is None:
    # budget set as a function of costs and target protection fraction
    budget = np.sum(graph_cost) * protection_fraction



def get_r_seeds_dict(seed):
    return {'rnd_sensitivity': 123 + seed,
                    'rnd_growth': 124 + seed,
                    'rnd_config_policy': 125 + seed,
                    'rnd_k_species': 126 + seed
                    }



# env_init GENERATOR
def generate_init_env(r_seeds_dict):
    if use_K_species:
        # species-specific carrying capacities
        rs_k = cn.get_rnd_gen(r_seeds_dict['rnd_k_species'])
        K_species = 2 + rs_k.random(n_species) * 100
        # make empirically rare species rare
        emp_threat = empirically_threatened[empirically_threatened < n_species]
        K_species[emp_threat] = np.sort(K_species)[:len(emp_threat)]
        K_species_3D = K_species[:, np.newaxis, np.newaxis] * graph_sdms * graph_suitability  # np.ones(h3d.shape)
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
        species_richness_init = sdm_utils.graph_to_grid(env2d.bioDivGrid.speciesPerCell(), reference_grid_pu)
        cn_util.plot_map(species_richness_init, z=reference_grid_pu_nan, nan_to_zero=False, vmax=max_species_richness,
                         cmap="YlGnBu", show=show_plots, title="Species richness (Natural state - FWD)",
                         outfile=os.path.join(results_wd, "species_richness_natural_FWD.png"), dpi=250)

        pop_density = sdm_utils.graph_to_grid(env2d.bioDivGrid.individualsPerCell(), reference_grid_pu)
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
        growth_rates = 2.

    mask_disturbance = (K_cells > 0).astype(int)
    disturbanceGenerator = cn.FixedEmpiricalDisturbanceGenerator(0)
    selective_disturbanceGenerator = cn.FixedEmpiricalDisturbanceGenerator(0)
    if zero_disturbance:
        zd = 0
    else: zd = 1
    init_disturbance = disturbanceGenerator.updateDisturbance(graph_disturbance * mask_disturbance * zd)
    init_selective_disturbance = selective_disturbanceGenerator.updateDisturbance(graph_selective_disturbance * mask_disturbance * 0)

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
                                  heuristic_policy="random",
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
    return env_init, config_init


envList = []
for i in range(BATCH_SIZE):
    r_seeds_dict = get_r_seeds_dict(SEED * i)
    env_train, config_train = generate_init_env(r_seeds_dict)
    if i == 0:
        env_train._verbose = True
    else:
        env_train._verbose = False
    env_train.rewardMode = reward
    env_train.iterations = STEPS
    env_train.budget = np.sum(graph_cost) * protection_fraction
    actions_per_step = 100
    env_train.reset_init_values()
    env_train.set_calc_reward(True)
    envList.append(copy.deepcopy(env_train))

cn.runBatchGeneticStrategyEmpirical(envList,
                                    epochs=1000,
                                    lr=0.5,
                                    lr_adapt=0.01,
                                    temperature=10,
                                    max_workers=0,
                                    outfile="empirical_trainingNN_sp_risk_protect.log",
                                    obsMode=1,
                                    observe_error=0,
                                    running_reward_start=-1000,
                                    eps_running_reward=0.5,
                                    sigma=0.2,
                                    wNN=None,
                                    n_NN_nodes=[1, 0],
                                    increase_temp=1 / 10,
                                    resolution=size_protection_unit,
                                    max_temperature=1000,
                                    sp_threshold_feature_extraction=1,
                                    wd_output=results_wd,
                                    actions_per_step=actions_per_step
                                    )

#### END TRY TRAINING


