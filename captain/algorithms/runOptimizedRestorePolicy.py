# TODO fix imports
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
import numpy as np

np.set_printoptions(suppress=True, precision=3)  # prints floats, no scientific notation
from ..biodivsim.StateInitializer import *
from .reinforce import RichProtectActionAdaptor, RestoreRichStateAdaptor
from ..agents.policy import PolicyRestore, get_NN_model_prm, PolicyNN, PolicyRestoreUpdateFreq
from .env_setup import *
from .marxan_setup import *
from ..plot.plot_env import plot_env_state
from ..plot.plot_features import plot_features
from ..biodivsim.BioDivEnv import *
from ..agents.state_monitor import extract_features_restore, get_feature_restore_indx, alter_init_grid, get_mean_grid_value_quadrant
import seaborn as sns
from ..biodivsim.SpeciesRiskClass import *

DEBUG=0
ADD_ERROR=0
PLOT = False
# if set to 1: each step means 1 action and 1 bioDivGrid.step
# if set to 5: each step means 1 action and bioDivGrid.step only occurs every 5 steps, i.e.
# each 5 actions for each bioDivGrid.step

class ConfigOptimPolicy():
    def __init__(self,
                 rnd_seed=1234,
                 # TODO: fix obsMode options
                 obsMode=0,  # 0: random, 1: full monitor, 2: citizen-science, 3: one-time, 4: value, 5: area
                 feature_update_per_step=True,
                 steps=10,
                 simulations=100,
                 observePolicy=1,  #  0: NO-OBSERVE-UPDATE 1: ORACLE 2: PROTECTATONCE
                 observe_error=0,
                 disturbance=4,
                 degrade_steps=3,
                 initial_disturbance=0.75,
                 edge_effect=1,
                 protection_cost=1,
                 n_nodes=[2, 2],
                 random_sim=0,  # "0: fixed (replicable) simulations; 1: random; 2: fixed training, seq pickle"
                 rewardMode=0,  # "0: species loss; 1: sp value; 2: protected area"; 3: PD loss (not yet tested)
                 obs_error=0,  # "Amount of error in species counts (feature extraction)"
                 use_true_natural_state=True,
                 resolution=np.array([5, 5]),
                 grid_size=50,
                 budget=0.11,
                 max_fraction_protected=1,
                 dispersal_rate=0.1,  # TODO: check if this can also be a vector per species
                 growth_rates=[0.1],  # can be 1 values (list of 1 item) or or one value per species
                 use_climate=3,  # "0: no climate change, 1: climate change, 2: climate disturbance,
                 # 3: climate change + random variation"
                 climate_change_magnitude=0.05,
                 peak_anomaly=2,
                 rnd_alpha=0,  # (st.dev of sp.-specific fluctuation in mortality (if 'by_species' ==1 in SimpleGrid)
                 dist_dependent_dispersal=0,
                 outfile="policy_output.log",
                 log_all_steps=False,
                 save_pkls=False,  # 0) no, 1) save pickle file at each step
                 # model settings
                 trained_model=None,  # if None: random policy
                 temperature=100,
                 deterministic_policy=1,  # 0: random policy (altered by temperature);
                 # 1: deterministic policy (overrides temperature)
                 wd="data_dependencies/pickles",
                 burnin=0,  # skip the first n. epochs (generally not needed)
                 load_best_epoch=0,  # 0: load last epoch; 1: load best epoch (post burnin)
                 sp_threshold_feature_extraction=1,
                 start_protecting=3,
                 plot_sim=False,
                 plot_species=[],
                 wd_output="",
                 grid_h = None,  # 3D hist of species (e.g. empirical)
                 distb_objects = None,  # list of distb_obj, selectivedistb_obj
                 return_env=False,
                 ext_risk_class=None,
                 evolve_rl_status=True,
                 max_K_multiplier=2,
                 suitability=None,
                 cost_layer=None,
                 initial_rl_status=None,
                 actions_per_step=1,
                 heuristic_policy=None,
                 minimize_policy=False,
                 use_empirical_setup=False,
                 max_protection_level=1,
                 dynamic_print=False,
                 precomputed_dispersal_probs=None,
                 use_small_grid=False,
                 sensitivities = None,
                 ext_risk_obj = None, # replaces ext_risk_class if provided
                 K_species_3D=None,
                 pre_steps = 0
                 ):
        self.seed = rnd_seed
        self.obsMode = obsMode
        self.feature_update_per_step = feature_update_per_step
        self.steps = steps
        self.simulations = simulations
        self.observePolicy = observePolicy
        self.observe_error = observe_error
        self.disturbance_mode = disturbance
        self.degrade_steps = degrade_steps
        self.initial_disturbance = initial_disturbance
        self.edge_effect = edge_effect
        self.use_protection_cost = protection_cost
        self.n_nodes = n_nodes
        self.random_sim = random_sim
        self.rewardMode = rewardMode
        self.obs_error = obs_error
        self.use_true_natural_state = use_true_natural_state
        self.resolution = resolution
        self.grid_size = grid_size
        self.budget = budget
        self.max_fraction_protected = max_fraction_protected
        self.dispersal_rate = dispersal_rate
        self.growth_rates = growth_rates
        self.use_climate = use_climate
        self.climate_change_magnitude = climate_change_magnitude
        self.peak_anomaly = peak_anomaly
        self.rnd_alpha_species = rnd_alpha
        self.disturbance_dep_dispersal = dist_dependent_dispersal
        self.outfile = outfile
        self.log_all_steps = log_all_steps
        self.save_pkls = save_pkls
        self.trained_model = trained_model
        self.temperature = temperature
        self.deterministic_policy = deterministic_policy
        self.wd = wd
        self.burnin = burnin
        self.load_best_epoch = load_best_epoch
        self.sp_threshold_feature_extraction = sp_threshold_feature_extraction
        self.start_protecting = start_protecting
        self.plot_sim = plot_sim
        self.plot_species = plot_species
        self.wd_output = wd_output
        self.grid_h = grid_h
        self.distb_objects = distb_objects
        self.return_env = return_env
        self.ext_risk_class = ext_risk_class
        self.ext_risk_obj = ext_risk_obj
        self.max_K_multiplier = max_K_multiplier
        self.suitability = suitability
        self.cost_layer = cost_layer
        self.initial_rl_status = initial_rl_status
        self.actions_per_step = actions_per_step
        self.heuristic_policy = heuristic_policy
        self.minimize_policy = minimize_policy
        self.evolve_rl_status = evolve_rl_status
        self.use_empirical_setup = use_empirical_setup
        self.max_protection_level = max_protection_level
        self.dynamic_print = dynamic_print
        self.precomputed_dispersal_probs = precomputed_dispersal_probs
        self.use_small_grid = use_small_grid
        self.sensitivities = sensitivities
        self.K_species_3D = K_species_3D
        self.pre_steps = pre_steps

    def set_ext_risk_class(self, ex: ExtinctionRisk):
        self.ext_risk_class = ex

    def set_runmode(self, runMode, wNN):
        self.runMode= runMode
        self.wNN = wNN

    def set_climate(self, climate_obj, climate_disturbance):
        self.climate_obj = climate_obj
        self.climate_as_disturbance = climate_disturbance
    def set_rnd_policy(self, p):
        self.rnd_policy = p

    def set_log_file_steps(self, s):
        self.log_file_steps = s

class EvolveEnv():
    def __init__(self, state_adaptor, action_adaptor, config):
        self._state_adaptor = state_adaptor
        self._action_adaptor = action_adaptor
        self._rewards = []
        self._outfile = config.outfile.split(".log")[0]
        self._count_rep = 0
        self._random_policy = config.rnd_policy
        self._save_pkls = config.save_pkls
        self._log_file_steps = config.log_file_steps
        if self._log_file_steps != "":
            with open(self._log_file_steps, "w") as f:
                writer = csv.writer(f, delimiter="\t")
                l = [
                    "simulation",
                    "step",
                    "protected_cells",
                    "species",
                    "value",
                    "pd",
                    "disturbance",
                ]
                writer.writerow(l)
        self._deterministic_policy = config.deterministic_policy
        self._plot_sim = config.plot_sim
        self._plot_species = config.plot_species
        self._sim_count = 0
        self._degrade_steps = config.degrade_steps
        self._initial_disturbance = config.initial_disturbance  # mean overall disturbance
        self._actions_per_step = config.actions_per_step
        self._seed = config.seed
        self._wd_output = config.wd_output
        self.species_list_plot = []
        self.plot_variables = ['diversity', 'density', 'rank-abundance', 'phylogeny',
                               'disturbance', 'value', 'carbon', 'cost', 'time-series',
                               'risk-label', 'protected-threatened', 'metrics']
        self.heuristic_policy = config.heuristic_policy
        if self.heuristic_policy is not None:
            self._deterministic_policy = True # overwrites config settings
        if self.heuristic_policy == "random":
            self._deterministic_policy = False
        self.minimize_policy = config.minimize_policy
        self.use_empirical_setup = config.use_empirical_setup
        self.max_protection_level = config.max_protection_level
        self.dynamic_print = config.dynamic_print
        self.feature_update_per_step = config.feature_update_per_step
        self._probs_heuristic_policy = None

    def set_init_disturbance(self, d):
        self._init_disturbance = d

    def set_init_rl_status(self, r):
        self._init_rl_status = r

    def select_action(self, state, info, policy, probs=None):
        state = self._state_adaptor.adapt(state, info)
        if probs is None:
            probs = policy.probs(state)
        if self._deterministic_policy:
            action = np.argmax(probs)
        else:
            action = np.random.choice(policy.num_output, 1, p=probs)
        return self._action_adaptor.adapt(action.item())

    def get_probs_heuristic_policy(self,
                                   policy: PolicyNN,
                                   env: BioDivEnv):
        if self._probs_heuristic_policy is None:
            # print("self.heuristic_policy", self.heuristic_policy)
            if self.heuristic_policy == "random":
                prob = np.ones(policy._num_output)
            elif self.heuristic_policy == "cheapest":
                costs = env.getProtectCostQuadrant()
                prob = 1 / (costs + np.min(costs[costs > 0]))
            elif self.heuristic_policy == "most_biodiverse":
                prob = get_mean_grid_value_quadrant(env.bioDivGrid.speciesPerCell(),
                                                    quandrant_grid_indx=env._quandrant_grid_indx)
            elif self.heuristic_policy == "most_natural_biodiversity":
                prob = get_mean_grid_value_quadrant(env.grid_obj_previous.speciesPerCell(),
                                                    quandrant_grid_indx=env._quandrant_grid_indx)
            elif self.heuristic_policy == "most_natural_carbon":
                prob = get_mean_grid_value_quadrant(env.grid_obj_previous.getCarbonValue_cell(),
                                                    quandrant_grid_indx=env._quandrant_grid_indx)
            elif self.heuristic_policy == "most_current_carbon":
                prob = get_mean_grid_value_quadrant(env.grid_obj_most_recent.getCarbonValue_cell(),
                                                    quandrant_grid_indx=env._quandrant_grid_indx)
            elif self.heuristic_policy == "highest_MSA":
                prob = calc_MSA_from_grid(env.grid_obj_previous.h, env.grid_obj_most_recent.h,
                                          quandrant_grid_indx=env._quandrant_grid_indx)
            elif self.heuristic_policy == "highest_STAR_t":
                prob, _ = calc_STAR_from_grid(env.grid_obj_previous.h, env.grid_obj_most_recent.h,
                                              quandrant_grid_indx=env._quandrant_grid_indx,
                                              sp_natural_range=env.grid_obj_previous.geoRangePerSpecies(),
                                              sp_ext_risk=env.getExtinction_risk_labels())
            elif self.heuristic_policy == "highest_STAR_r":
                _, prob = calc_STAR_from_grid(env.grid_obj_previous.h, env.grid_obj_most_recent.h,
                                              quandrant_grid_indx=env._quandrant_grid_indx,
                                              sp_natural_range=env.grid_obj_previous.geoRangePerSpecies(),
                                              sp_ext_risk=env.getExtinction_risk_labels())

            self._probs_heuristic_policy = prob + 0
        else:
            prob = self._probs_heuristic_policy + 0
        # apply minimization policy
        if self.minimize_policy:
            prob = 1 / (prob + 0.00001)

        # remove e.g. sea
        m = get_mean_grid_value_quadrant(env.bioDivGrid._K_cells > 0,
                                         quandrant_grid_indx=env._quandrant_grid_indx)
        prob[m == 0] = 0

        # remove already protected/restored
        prob[np.array(env.protected_quadrants).astype(int)] = 0
        # normalize probabilities
        prob = prob / np.sum(prob)
        return prob



    def save_grid(self, env, step=0):
        # TODO: check this (fix imports or remove )
        filename = self._outfile + "_%s_step%s_.pkl" % (self._count_rep, step)
        print("Saving pickle file:", filename)
        SaveObject(env, filename)

    def log_steps(self, env, step=0):
        with open(self._log_file_steps, "a") as f:
            writer = csv.writer(f, delimiter="\t")
            info = env._getInfo()
            l = [
                self._sim_count,
                step,
                info["NumberOfProtectedCells"],
                info["ExtantSpecies"],
                info["ExtantSpeciesValue"],
                info["ExtantSpeciesPD"],
                np.mean(env.bioDivGrid.disturbance_matrix),
            ]
            writer.writerow(l)


class EVOLUTIONBoltzmannPredict(EvolveEnv):
    """
     outfile, rnd_policy=0, save_pkls=0, log_file_steps="",
                 deterministic_policy=0, plot_sim=0, plot_species=[], degrade_steps=5, initial_disturbance=0.79,
                 actions_per_step=1,
                 wd_output="", seed=None
    """
    def __init__(self, state_adaptor, action_adaptor, config):
        super().__init__(state_adaptor, action_adaptor, config)

        if config.use_empirical_setup:
            self._init_disturbance = None
            self._init_rl_status = None
            self._evolve_rl_status = config.evolve_rl_status
            self._ext_risk_class = config.ext_risk_class
            self._ext_risk_obj = config.ext_risk_obj
            self._max_K_multiplier = config.max_K_multiplier
            self._suitability = config.suitability
            self._cost_layer = config.cost_layer
            self._pre_steps = config.pre_steps

    def run_episode(self, env: BioDivEnv, policy: PolicyNN):
        if self.use_empirical_setup:
            # set natural carrying capacity to empirical one times a factor
            # if factor == 1: will lose diversity because mortality is high at carrying capacity
            # could multiply h3d by a value to have "individuals" instead of presence/absence
            # print("K DEBUG - ini pop 0 ", env.bioDivGrid.individualsPerSpecies())
            K_cells = np.einsum('sxy -> xy', env.bioDivGrid.h)
            env.bioDivGrid.reset_K_max(K_cells * self._max_K_multiplier)
            env.set_calc_reward(False)  # avoid computing reward (time-consuming)
            env.set_max_protection_level(self.max_protection_level)
            # print("K DEBUG - ini pop 1 ", env.bioDivGrid.individualsPerSpecies())

            # add habitat suitability
            if self._suitability is None:
                h3d = env.bioDivGrid.h + 0
                suitab = h3d * 0 + 0.01
                suitab[h3d >= 1] = 1  # only allow dispersal where suitable niche
                env.bioDivGrid.set_habitat_suitability(suitab)
            else:
                env.bioDivGrid.set_habitat_suitability(self._suitability)

            env.update_previous_observation = True
            # fix 'natural' baseline for feature extraction
            env.observe()
        else:
            # # init env (randomize and re-load pickle file)
            if self._seed:
                r = self._seed + self._sim_count
            else:
                r = 1000 + self._sim_count
            list_species_values = init_sp_values(env.bioDivGrid._n_species,
                                                 seed=r,
                                                 grid_size=env.bioDivGrid.length)
            # env.list_species_values = list_species_values
            env.list_species_values_init = list_species_values
            env.update_previous_observation = True
            _ = env.reset(fullInfo=True)
            # PRE EVOLVE ENV
            # add heterogeneity in carrying capacity
            env.heterogeneous_carrying_capacity(baseline_K=0.3,
                                                species_cell_K=False,
                                                seed=r+1)

            # fix 'natural' baseline for feature extraction
            env.observe()

            # TODO: expose this
            if ADD_ERROR:
                #--- ADD ERROR in knowledge of the natural state
                """
                REPLACE with 'empirical' natural state and use this to alter the observation of the current state?
                """
                self.per_species_obs_err = 10
                self.dd_fraction = 0.3
                alter_init_grid(env.grid_obj_previous._h,
                                per_species_obs_err=self.per_species_obs_err,
                                dd_fraction=self.dd_fraction
                                )



                h_tmp = env.grid_obj_previous._h + 0

                # add over and under estimation of species
                multi = np.exp(np.random.uniform(
                    np.log(0.1), np.log(10), h_tmp.shape[0]
                ))
                h_tmp = np.einsum('sxy, s -> sxy', h_tmp, multi)
                # probability based on abundance -> rare more difficult to predict
                bias_steepness = 1 # (smaller than 1 flattens probs and increases missing data)
                p_unobserved = (1 / (1 + h_tmp)) ** bias_steepness # (species, cell, cell)
                rr = np.random.random(h_tmp.shape)
                h_tmp[rr < p_unobserved] = 0
                env.set_grid_obj_h_previous(h_tmp)
                # missing species in Red List assessment
                rr = np.random.random(h_tmp.shape[0])
                dd_fraction = 0.3 # 30% unassessed species
                DD_species_indx = np.arange(h_tmp.shape[0])[rr < dd_fraction] #
                # print("DD_species_indx", DD_species_indx, len(DD_species_indx))
                # random status
                # rnd_status = np.random.randint(0, env.species_risk_criteria.n_labels, len(DD_species_indx))
                # all LC
                rnd_status = np.ones(len(DD_species_indx)) * (env.species_risk_criteria.n_labels - 1)
                p_dd_class = np.array([0.6, 0.4, 0.3, 0.1, 0.1])
                # print("rnd_status", rnd_status)

        #---

        # from now on do not update 'previous_observation'
        env.update_previous_observation = False
        env.dynamic_print = self.dynamic_print

        if DEBUG:
            print(r, list_species_values[0], np.sum(env.bioDivGrid.h))

        if self._plot_sim:
            plot_env_state(env, wd=self._wd_output, variables=self.plot_variables,
                           species_list=self.species_list_plot,
                           outfile=self._outfile + "_sim%s.natural_state" % self._sim_count)

            f = extract_features_restore(grid_obj=env.grid_obj_most_recent,
                                         grid_obj_previous=env.grid_obj_previous,
                                         quadrant_resolution=env.resolution,
                                         current_protection_matrix=env.bioDivGrid.protection_matrix,
                                         species_threat_label=env.getExtinction_risk_labels(),
                                         n_threat_labels=env.species_risk_criteria.n_labels,
                                         quandrant_grid_indx=env._quandrant_grid_indx,
                                         cost_quadrant=env.getProtectCostQuadrant(),
                                         budget=env.budget,
                                         normalize=False
                                         )

            plot_features(env, f, self._wd_output,
                          self._outfile + "_sim%s.features-natural_%s" % (self._sim_count, str(env.currentIteration)))

        # --- 2. fast-forward disturbance and environment degradation ---#
        if self.use_empirical_setup:
            dm_tmp = self._init_disturbance + 0
            env.bioDivGrid.setDisturbanceMatrix(dm_tmp)
            env.bioDivGrid.setSelectiveDisturbanceMatrix(dm_tmp)
            # degrade environment
            # print("K DEBUG - ini pop 2 ", env.grid_obj_previous.individualsPerSpecies())
            # print("K DEBUG - ini pop 3", env.bioDivGrid.individualsPerSpecies())
            env.fast_forward(self._degrade_steps,
                             disturbance_effect_multiplier=5,
                             verbose=False,
                             skip_dispersal=True,
                             additional_steps=self._pre_steps)
            # print("K DEBUG - ini pop 4", env.bioDivGrid.individualsPerSpecies())

            if PLOT:
                sns.heatmap(env.bioDivGrid.disturbance_matrix, cmap="RdYlGn_r")
                plt.show()
                sns.heatmap(env.bioDivGrid._K_max, cmap="Greens")
                plt.show()
                sns.heatmap(env.bioDivGrid.speciesPerCell(), cmap="coolwarm")
                plt.show()
                print(env.disturbanceGenerator)

            # set Ext_risk criteria
            if self._ext_risk_obj is not None:
                self._ext_risk_class = self._ext_risk_obj.__class__
                if self._init_rl_status is not None:
                    print("WARNING: Ignoring self._init_rl_status and using ext_risk_obj._evolve_rl_status")
                ext_obj = self._ext_risk_class(natural_state=env.grid_obj_previous,
                                               current_state=env.bioDivGrid,
                                               starting_rl_status=self._ext_risk_obj.starting_rl_status,
                                               evolve_status=self._ext_risk_obj.evolve_status,
                                               relative_pop_thresholds=self._ext_risk_obj.relative_pop_thresholds,
                                               epsilon=self._ext_risk_obj.epsilon,
                                               sufficient_protection=self._ext_risk_obj.sufficient_protection,
                                               pop_decrease_threshold=self._ext_risk_obj._pop_change_threshold,

                                               )
            else:
                ext_obj = self._ext_risk_class(natural_state=env.grid_obj_previous,
                                               current_state=env.bioDivGrid,
                                               starting_rl_status=self._init_rl_status,
                                               evolve_status=self._evolve_rl_status)



            env.set_species_risk_criteria(ext_obj)
            # env.species_risk_criteria.set_init_pop_sizes(env.bioDivGrid.individualsPerSpecies())

            print("self._ext_risk_class", self._ext_risk_class.__class__)
            print("self._init_rl_status", self._init_rl_status)
            print("_init_sp_ext_risk", env._init_sp_ext_risk)

        else:
            if self._initial_disturbance is not None:
                distb_obj, _ = get_disturbance(8,seed=r+2)
                dm_tmp = env.bioDivGrid.disturbance_matrix + 0
                max_c = 0
                while np.mean(dm_tmp) <= self._initial_disturbance:
                    # update disturbance until target reached
                    dm_tmp = distb_obj.updateDisturbance(dm_tmp)
                    max_c += 1
                    if max_c > 100: break

                env.bioDivGrid.setDisturbanceMatrix(dm_tmp)
            # env.bioDivGrid.setSelectiveDisturbanceMatrix(dm_tmp)
            # env.evolve(self._degrade_steps, verbose=True)  # degrade environment
            env.fast_forward(self._degrade_steps,
                             disturbance_effect_multiplier=5,
                             verbose=False,
                             additional_steps=self._pre_steps)

        # set current env as starting state
        env.reset_init_values()
        env.observe()
        if self._plot_sim:
            plot_env_state(env, wd=self._wd_output, variables=self.plot_variables,
                           species_list=self.species_list_plot,
                           outfile=self._outfile + "_sim%s.starting_state" % self._sim_count)

            f = extract_features_restore(grid_obj=env.grid_obj_most_recent,
                                         grid_obj_previous=env.grid_obj_previous,
                                         quadrant_resolution=env.resolution,
                                         current_protection_matrix=env.bioDivGrid.protection_matrix,
                                         species_threat_label=env.getExtinction_risk_labels(),
                                         n_threat_labels=env.species_risk_criteria.n_labels,
                                         quandrant_grid_indx=env._quandrant_grid_indx,
                                         cost_quadrant=env.getProtectCostQuadrant(),
                                         budget=env.budget,
                                         normalize=False
                                         )

            plot_features(env, f, self._wd_output,
                          self._outfile + "_sim%s.features_%s" % (self._sim_count, str(env.currentIteration)))

        state = env._enrichObs()
        info = env._getInfo()

        init_threat_levels = env.risk_label_counts(normalize=True)
        init_carbon = info["TotalCarbon"] # np.sum(self.bioDivGrid.getCarbonValue_cell()) / self._init_total_carbon

        if DEBUG:
            print("NEW h", np.sum(env.bioDivGrid.h))
        ep_reward = 0

        # for t in range(1, env.iterations):  # Don't infinite loop while learning
        counter = 0
        while True:
            # TODO: expose this
            if ADD_ERROR:
                if counter == 0:
                    pp = p_dd_class[state["species_threat_label"]]
                    state["species_threat_label"][pp > np.random.random(len(pp))] = 4
                    err_status = state["species_threat_label"] + 0 #* 0 + 4
                    print("state[species_threat_label]:", state["species_threat_label"])
                    # state["species_threat_label"][DD_species_indx] = rnd_status
                state["species_threat_label"] = err_status + 0

            skip_env_step = counter % self._actions_per_step != 0

            if self.heuristic_policy is None:
                action = self.select_action(state, info, policy)
            else:
                probs = self.get_probs_heuristic_policy(policy, env)
                action = self.select_action(state, info, policy, probs=probs)
            state, reward, done, info = env.step(action, skip_env_step=skip_env_step)
            if skip_env_step is False:
                env.species_risk_criteria.update_pop_sizes(env.bioDivGrid)
                # decide if update features or not
                if self.feature_update_per_step:
                    policy.reset_lastObs(None)
                # print("skip_env_step", skip_env_step, counter, self._actions_per_step)

            # state here is richObs in BioDivEnv which is lastObs.stats_quadrant + timeSince last observe
            self._rewards.append(reward)
            ep_reward += reward
            counter += 1
            if done:
                break

            if self._plot_sim:
                f = extract_features_restore(grid_obj=env.grid_obj_most_recent,
                                             grid_obj_previous=env.grid_obj_previous,
                                             quadrant_resolution=env.resolution,
                                             current_protection_matrix=env.bioDivGrid.protection_matrix,
                                             species_threat_label=env.getExtinction_risk_labels(),
                                             n_threat_labels=env.species_risk_criteria.n_labels,
                                             quandrant_grid_indx=env._quandrant_grid_indx,
                                             cost_quadrant=env.getProtectCostQuadrant(),
                                             budget=env.budget,
                                             normalize=False,
                                             get_protected_species_list=False
                                             )

                # plot log probabilities
                if self.heuristic_policy is None:
                    state_tmp = self._state_adaptor.adapt(state, info)
                    probs = np.log(1 + policy.probs(state_tmp))
                    # print("\n\nf.stats_quadrant", f.stats_quadrant)
                    # print("\n\nf.stats_quadrant SHAPE", f.stats_quadrant.shape)
                    f.feature_names = np.append(f.feature_names, "prob")
                    f.stats_quadrant = np.hstack((f.stats_quadrant, probs.reshape(len(probs), 1)))
                    print("f.feature_names", f.feature_names)


                plot_env_state(env, wd=self._wd_output, variables=self.plot_variables,
                               species_list=self.species_list_plot,
                               outfile=self._outfile + "_sim%s.restored_state" % self._sim_count)


                plot_features(env, f, self._wd_output,
                              self._outfile + "_sim%s.features_%s" % (self._sim_count, str(env.currentIteration)))

        # print(np.array(env.history))
        self._sim_count += 1

        info['init_threat_levels'] = init_threat_levels
        info['init_carbon'] = init_carbon

        return info, ep_reward





class EVOLUTIONEmpiricalPredict(EvolveEnv):
    """
     outfile, rnd_policy=0, save_pkls=0, log_file_steps="",
                 deterministic_policy=0, plot_sim=0, plot_species=[], degrade_steps=5, initial_disturbance=0.79,
                 actions_per_step=1,
                 wd_output="", seed=None
    """
    def __init__(self, state_adaptor, action_adaptor, config,
                 feature_update_per_step=True,
                 actions_per_step=1):
        # super().__init__(state_adaptor, action_adaptor, config)

        self.feature_update_per_step = config.feature_update_per_step
        self._actions_per_step = actions_per_step
        self._state_adaptor = state_adaptor
        self._action_adaptor = action_adaptor
        self.heuristic_policy = config.heuristic_policy
        self._deterministic_policy = config.deterministic_policy
        self._rewards = []
        self.minimize_policy = config.minimize_policy
        self._probs_heuristic_policy = None

    def run_episode(self, env, policy):
        print("\n\nEVOLUTIONEmpiricalPredict.run_episode")
        state = env._enrichObs()
        info = env._getInfo()
        if DEBUG:
            print("NEW h", np.sum(env.bioDivGrid.h))
            print(" env.risk_label_counts(normalize=True)",  env.risk_label_counts(normalize=True))

        ep_reward = 0
        # for t in range(1, env.iterations):  # Don't infinite loop while learning
        counter = 0
        while True:
            # print("policy.reset_lastObs(None)", env.currentIteration,
            #       env.bioDivGrid._counter, self._actions_per_step, counter)
            skip_env_step = counter % self._actions_per_step != 0

            if skip_env_step:
                env.set_calc_reward = False
            else:
                env.set_calc_reward = True
            if self.heuristic_policy is None:
                action = self.select_action(state, info, policy)
            else:
                probs = self.get_probs_heuristic_policy(policy, env)
                action = self.select_action(state, info, policy, probs=probs)
            state, reward, done, info = env.step(action, skip_env_step=skip_env_step)
            if skip_env_step is False:
                env.species_risk_criteria.update_pop_sizes(env.bioDivGrid)
                # decide if update features or not
                if self.feature_update_per_step:
                    policy.reset_lastObs(None)
                    self._probs_heuristic_policy = None


            counter += 1
            self._rewards.append(reward)
            ep_reward += reward

            if done:
                break

        # print(np.array(env.history))
        if DEBUG:
            print(" env.risk_label_counts(normalize=True)",  env.risk_label_counts(normalize=True))



RunnerInput = collections.namedtuple("RunnerInput", ("env", "policy", "runner"))
EvolutionRunnerInput = collections.namedtuple(
    "EvolutionRunnerInput", ("env", "policy", "runner", "noise")
)


epoch_data_head = [
    "simulation",
    "reward",
    "protected_cells",
    "budget_left",
    "time_last_protect",
    "avg_cost",
    "extant_sp",
    "extant_sp_value",
    "extant_sp_pd"
]
epoch_data_head = epoch_data_head + ["tot_carbon", 'CR/EX', 'EN', 'VU', 'NT', 'LC']

epoch_data_head = epoch_data_head + ["tot_carbon_0", 'CR/EX_0', 'EN_0', 'VU_0', 'NT_0', 'LC_0']



def _runOptimPolicySetup(config: ConfigOptimPolicy):

    if config.grid_h is not None:
        gridInitializer = EmpiricalStateInit(config.grid_h)
        # set empirical carrying capacity
        K_cells = np.einsum('sxy -> xy', config.grid_h)
        K_max = np.max(K_cells)
        n_species = config.grid_h.shape[0]
        n_cells = config.grid_h.shape[1]
        if config.use_empirical_setup is False:
            print("\n\nSetting config.use_empirical_setup to True!\n\n")
            config.use_empirical_setup = True
    else:
        if config.sim_file is not None:
            gridInitializer = PickleInitializer(config.sim_file)
            rnd_disturbance_init = config.disturbance_mode # TODO: This or -1?
        else:
            gridInitializer = PickleInitializerSequential(
                pklfolder=config.wd, verbose=True, pklfile_i=config.pickle_num
            )
        init_data = gridInitializer.getInitialState(1, 1, 1)
        n_cells = init_data.shape[1]
        n_species = init_data.shape[0]
        K_max = np.einsum("xyz -> yz", init_data)[0][0]
        empirical_evolve_env = False

    alpha = 0.01
    OUTPUT = (n_cells ** 2) / (config.resolution[0] * config.resolution[1])
    if OUTPUT % int(OUTPUT) != 0:
        sys.exit("\n\nResolution not allowed!\n\n")
    else:
        OUTPUT = int(OUTPUT)
        print("Number of protection units: ", OUTPUT)

    if config.distb_objects is None:
        distb_obj, selectivedistb_obj = get_disturbance(config.disturbance_mode, config.seed)
        initial_disturbance_runner = config.initial_disturbance + 0
    else:
        distb_obj = config.distb_objects[0]
        selectivedistb_obj = config.distb_objects[1]
        initial_disturbance_runner = None


    if config.random_sim:
        disturbance_sensitivity = np.zeros(n_species) + np.random.random(n_species)
        selective_sensitivity = np.random.beta(0.2, 0.7, n_species)
        climate_sensitivity = np.random.beta(2, 2, n_species)
        list_species_values = []
    else:
        if config.sensitivities is None:
            (
                disturbance_sensitivity,
                selective_sensitivity,
                climate_sensitivity,
            ) = init_sp_sensitivities(n_species, seed=config.seed)
        else:
            disturbance_sensitivity = config.sensitivities['disturbance_sensitivity']
            selective_sensitivity = config.sensitivities['selective_sensitivity']
            climate_sensitivity = config.sensitivities['climate_sensitivity']

        list_species_values = init_sp_values(n_species, seed=config.seed, grid_size=n_cells)

    print("disturbance_sensitivity in _runOptimPolicySetup", disturbance_sensitivity)
    # species loss is calculated in BioDivEnv: `reward = self.bioDivGrid.numberOfSpecies() - self.n_extant`
    if config.obsMode == 0:
        # the only feature is "check already protected", i.e. RANDOM POLICY
        config.set_rnd_policy(1)
    elif config.obsMode == 6:
        config.set_rnd_policy(2)
        # check deltaVC feature, i.e. HEURISTIC POLICY
    else:
        config.set_rnd_policy(0)
        if config.wNN is None:
            sys.exit("Please provide trained model or use obsMode [0,6]")

    feature_set = config.rewardMode  # "all" # rewardMode
    num_features = len(get_feature_restore_indx(mode=config.rewardMode))
    print("num_features", num_features)
    print(get_feature_restore_indx(mode=feature_set))
    [
        num_output,
        num_meta_features,
        nodes_layer_1,
        nodes_layer_2,
        nodes_layer_3,
        n_prms,
    ] = get_NN_model_prm(num_features, config.n_nodes, OUTPUT)

    if config.wNN is None:
        coeff_features = np.ones(n_prms)
        coeff_meta_features = np.zeros(num_meta_features)
        if config.rnd_policy == 1:
            print("Running with random policy")
        else:
            print("Running with heuristic policy")
    else:
        coeff_features = config.wNN[:-num_meta_features]
        coeff_meta_features = config.wNN[-num_meta_features:]

    policy = PolicyRestoreUpdateFreq(num_features,
                                     num_meta_features,
                                     num_output,
                                     coeff_features,
                                     coeff_meta_features,
                                     temperature=config.temperature,
                                     mode=config.obsMode,
                                     feature_set=feature_set,
                                     observe_error=config.observe_error,
                                     use_true_natural_state=config.use_true_natural_state,
                                     nodes_l1=nodes_layer_1,
                                     nodes_l2=nodes_layer_2,
                                     nodes_l3=nodes_layer_3,
                                     sp_threshold=config.sp_threshold_feature_extraction,
                                     )

    state_adaptor = RestoreRichStateAdaptor()
    action_adaptor = RichProtectActionAdaptor(n_cells, config.resolution)
    # init out file
    outfile_w_path = os.path.join(config.wd_output, config.outfile)

    with open(outfile_w_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(epoch_data_head)
    # TODO end refactor
    if config.log_all_steps:
        config.set_log_file_steps(outfile_w_path + "_steps.log")
    else:
        config.set_log_file_steps("")

    evolutionRunner = EVOLUTIONBoltzmannPredict(state_adaptor,
                                                action_adaptor,
                                                config)
    evolutionRunner.set_init_disturbance(config.initial_disturbance)
    if config.use_empirical_setup:
        evolutionRunner.set_init_rl_status(
            config.initial_rl_status)  # if config.initial_rl_status -> RL init based on pop sizes

    # 'budget', 'gridInitializer', 'n_cells', 'n_species', 'alpha', 'K_max', 'dispersal_rate',
    # 'distb_obj', 'disturbance_sensitivity', 'selectivedistb_obj', 'selective_sensitivity', 'climate_obj', 'timeSteps',
    # 'runMode', 'worker_id', 'obsMode', 'use_protection_cost', 'random_training',
    # 'rnd_disturbance_init', 'rewardMode', 'list_species_values'

    # if random_sim == 1:
    envInput = EnvInput(config.budget,
                        gridInitializer,
                        n_cells,
                        n_species,
                        alpha,
                        K_max,
                        config.dispersal_rate,
                        distb_obj,
                        disturbance_sensitivity,
                        selectivedistb_obj,
                        selective_sensitivity,
                        config.climate_obj,
                        climate_sensitivity,
                        config.steps,
                        config.runMode,
                        0,
                        config.obsMode,
                        config.use_protection_cost,
                        config.random_sim,
                        config.disturbance_mode,
                        config.rewardMode,
                        list_species_values,
                        config.resolution,
                        config.climate_as_disturbance,
                        config.rnd_alpha_species,
                        config.disturbance_dep_dispersal,
                        config.max_fraction_protected,
                        config.edge_effect,
                        config.growth_rates,
                        config.start_protecting,
                        )

    return (
        envInput,
        n_species,
        n_cells,
        evolutionRunner,
        policy,
        distb_obj,
        selectivedistb_obj,
    )


def runOptimPolicy(config: ConfigOptimPolicy):

    init_obj = _runOptimPolicySetup(config)

    (envInput, n_species, n_cells,
         evolutionRunner, policy, distb_obj,
         selectivedistb_obj,) = init_obj


    print("=======================================")
    print("setup done! Running simulations...")
    print("=======================================")
    env = buildEnv(envInput, config)
    for epoch in range(config.simulations):
        print("=======================================")
        print(f"running simulation: {epoch}")
        print("=======================================")

        res = evolutionRunner.run_episode(env, policy)

        print("=======================================")
        print(f"epoch {epoch} summary")
        print("=======================================")
        print(f"policy coeff: {policy.coeff}")
        print(f"avg reward: {np.sum(res[1])}")
        print("budget left", res[0]["budget_left"])
        print("time last protect", res[0]["time_last_protect"])
        print("n. protected cells", res[0]["NumberOfProtectedCells"])
        # print("selected units:", env.protected_quadrants)

        avg_reward = np.sum(res[1])
        avg_budget_left = res[0]["budget_left"]
        avg_time_last_protect = res[0]["time_last_protect"]
        avg_protected_cells = res[0]["NumberOfProtectedCells"]
        avg_cost = res[0]["CostOfProtection"]
        avg_extant_sp = res[0]["ExtantSpecies"]
        avg_extant_sp_value = res[0]["ExtantSpeciesValue"]
        avg_extant_sp_pd = res[0]["ExtantSpeciesPD"]


        print(res[0]['init_threat_levels'])

        epoch_data = [
            epoch,
            avg_reward,
            avg_protected_cells,
            avg_budget_left,
            avg_time_last_protect,
            avg_cost,
            avg_extant_sp,
            avg_extant_sp_value,
            avg_extant_sp_pd,
            res[0]['TotalCarbon'],
            res[0]['CR/EX'],
            res[0]['EN'],
            res[0]['VU'],
            res[0]['NT'],
            res[0]['LC'],

            res[0]['init_carbon'],
        ] + list(res[0]['init_threat_levels'])


        outfile_w_path = os.path.join(config.wd_output, config.outfile)
        with open(outfile_w_path, "a") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(epoch_data)

    return env



def _run_policy_init(config: ConfigOptimPolicy):

    runMode = [RunMode.NOUPDATEOBS, RunMode.ORACLE, RunMode.PROTECTATONCE][
        config.observePolicy
    ]
    climate_disturbance = 0
    if config.use_climate:
        if config.use_climate == 1:
            climate_change = config.climate_change_magnitude
            from ..biodivsim.ClimateGenerator import (
                SimpleGradientClimateGenerator as ClimateGen,
            )

            CLIMATE_OBJ = ClimateGen(0, climate_change=climate_change)
        elif config.use_climate == 2:
            climate_disturbance = 1
            from ..biodivsim.ClimateGenerator import RegionalClimateGenerator as ClimateGen

            CLIMATE_OBJ = ClimateGen(0)
        elif config.use_climate == 3:  # global warming + random variation
            climate_change = config.climate_change_magnitude
            PEAK_ANOMALY = config.peak_anomaly
            from ..biodivsim.ClimateGenerator import (
                GradientClimateGeneratorRnd as ClimateGen,
            )

            CLIMATE_OBJ = ClimateGen(
                0, climate_change=climate_change, peak_anomaly=PEAK_ANOMALY
            )
    else:
        CLIMATE_OBJ = 0

    if config.trained_model is not None and config.trained_model != "heuristic":
        head = next(open(config.trained_model)).split()
        loaded_ws = np.loadtxt(config.trained_model, skiprows=np.max([1, config.burnin]))
        if config.load_best_epoch:
            selected_epoch = np.argmax(loaded_ws[:, head.index("reward")])
        else:
            selected_epoch = -1
        print(
            "Selected epoch",
            selected_epoch,
            loaded_ws[:, head.index("reward")][selected_epoch],
        )
        loadedW = loaded_ws[selected_epoch]

        feature_set = config.rewardMode  # "all" # rewardMode
        num_features = len(get_feature_restore_indx(mode=feature_set))
        print("num_features", num_features)
        print(get_feature_restore_indx(mode=feature_set))
        ind = [head.index(s) for s in head if "coeff_" in s]
        wNN = loadedW[
            np.min(ind) :
        ]  # remove first columns (reward, protected_cells, budget_left, time_last_protect, running_reward, etc.)
        running_reward_start = loadedW[head.index("running_reward")]

        print("wNN", wNN)
        print(running_reward_start)

    else:
        wNN = None
        if config.trained_model is None:
            n_nodes = [1, -1]
            obsMode = 0

    # # TODO: Why call this?
    # get_feature_indx(obsMode, print_obs_mode=True)

    return runMode, wNN, CLIMATE_OBJ, climate_disturbance


def run_restore_policy(config: ConfigOptimPolicy):
    runMode, wNN, climate_obj, climate_disturbance = _run_policy_init(config)
    config.set_runmode(runMode, wNN)
    config.set_climate(climate_obj, climate_disturbance)
    if config.ext_risk_class is None:
        config.set_ext_risk_class = ExtinctionRisk

    env = runOptimPolicy(config)

    if config.return_env:
        return env



def run_policy_empirical_env(config, env):

    grid_size = env.bioDivGrid.h.shape[1]
    runMode, wNN, climate_obj, climate_disturbance = _run_policy_init(config)
    config.set_runmode(runMode, wNN)
    state_adaptor = RestoreRichStateAdaptor()
    action_adaptor = RichProtectActionAdaptor(grid_size, config.resolution)

    # init out file
    outfile_w_path = os.path.join(config.wd_output, config.outfile)

    with open(outfile_w_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(epoch_data_head)
    # TODO end refactor
    if config.log_all_steps:
        config.set_log_file_steps(outfile_w_path + "_steps.log")
    else:
        config.set_log_file_steps("")

    # set up policy
    OUTPUT = int(env.num_quadrants)
    # species loss is calculated in BioDivEnv: `reward = self.bioDivGrid.numberOfSpecies() - self.n_extant`
    if config.obsMode == 0:
        # the only feature is "check already protected", i.e. RANDOM POLICY
        config.set_rnd_policy(1)
    elif config.obsMode == 6:
        config.set_rnd_policy(2)
        # check deltaVC feature, i.e. HEURISTIC POLICY
    else:
        config.set_rnd_policy(0)

    feature_set = config.rewardMode  # "all" # rewardMode
    num_features = len(get_feature_restore_indx(mode=config.rewardMode))
    print("num_features", num_features)
    print(get_feature_restore_indx(mode=feature_set))
    [
        num_output,
        num_meta_features,
        nodes_layer_1,
        nodes_layer_2,
        nodes_layer_3,
        n_prms,
    ] = get_NN_model_prm(num_features, config.n_nodes, OUTPUT)

    if config.trained_model is not None and config.trained_model != "heuristic":
        head = next(open(config.trained_model)).split()
        loaded_ws = np.loadtxt(config.trained_model, skiprows=np.max([1, config.burnin]))
        if config.load_best_epoch:
            selected_epoch = np.argmax(loaded_ws[:, head.index("reward")])
        else:
            selected_epoch = -1
        print(
            "Selected epoch",
            selected_epoch,
            loaded_ws[:, head.index("reward")][selected_epoch],
        )
        loadedW = loaded_ws[selected_epoch]

        feature_set = config.rewardMode  # "all" # rewardMode
        num_features = len(get_feature_restore_indx(mode=feature_set))
        print("num_features", num_features)
        print(get_feature_restore_indx(mode=feature_set))
        ind = [head.index(s) for s in head if "coeff_" in s]
        wNN = loadedW[
            np.min(ind) :
        ]

        print("wNN", wNN)
        coeff_features = wNN[:-num_meta_features]
        coeff_meta_features = wNN[-num_meta_features:]


    else:
        wNN = None
        if config.trained_model is None:
            n_nodes = [1, -1]
            obsMode = 0
        coeff_features = None
        coeff_meta_features = None



    policy = PolicyRestoreUpdateFreq(num_features,
                                     num_meta_features,
                                     num_output,
                                     coeff_features,
                                     coeff_meta_features,
                                     temperature=config.temperature,
                                     mode=config.obsMode,
                                     feature_set=feature_set,
                                     observe_error=config.observe_error,
                                     use_true_natural_state=config.use_true_natural_state,
                                     nodes_l1=nodes_layer_1,
                                     nodes_l2=nodes_layer_2,
                                     nodes_l3=nodes_layer_3,
                                     sp_threshold=config.sp_threshold_feature_extraction,
                                     quadrant_coords_list=env.quadrant_coords_list,
                                     )
    #
    evolutionRunner = EVOLUTIONEmpiricalPredict(state_adaptor, action_adaptor, config,
                                                actions_per_step=config.actions_per_step)

    evolutionRunner.run_episode(env, policy)
    # env.set_reward_weights({'carbon': 10, 'species_risk': 10})
    #
    # state = env._enrichObs()
    # info = env._getInfo()
    # if DEBUG:
    #     print("NEW h", np.sum(env.bioDivGrid.h))
    #     print(" env.risk_label_counts(normalize=True)", env.risk_label_counts(normalize=True))
    # ep_reward = 0
    #
    # # for t in range(1, env.iterations):  # Don't infinite loop while learning
    # counter = 0
    # while True:
    #     # print("policy.reset_lastObs(None)", env.currentIteration,
    #     #       env.bioDivGrid._counter, self._actions_per_step, counter)
    #     skip_env_step = counter % config.actions_per_step != 0
    #     if skip_env_step:
    #         env.set_calc_reward = False
    #     else:
    #         env.set_calc_reward = True
    #     action = self.select_action(state, info, policy)
    #     state, reward, done, info = env.step(action, skip_env_step=skip_env_step)
    #     if skip_env_step is False:
    #         env.species_risk_criteria.update_pop_sizes(env.bioDivGrid)
    #         # decide if update features or not
    #         if self.feature_update_per_step:
    #             policy.reset_lastObs(None)
    #
    #     # state here is richObs in BioDivEnv which is lastObs.stats_quadrant + timeSince last observe
    #     self._rewards.append(reward)
    #     ep_reward += reward
    #     counter += 1
    #     if done:
    #         break
    #
    # # print(np.array(env.history))
    # if DEBUG:
    #     print(" env.risk_label_counts(normalize=True)", env.risk_label_counts(normalize=True))
    # return self._rewards, info

    if config.return_env:
        return env



















