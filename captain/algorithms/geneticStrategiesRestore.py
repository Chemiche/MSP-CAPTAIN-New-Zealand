import csv
import os
import sys

# TODO fix imports
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
from ..biodivsim.BioDivEnv import BioDivEnv, Action, ActionType, RunMode
from ..agents.state_monitor import extract_features, get_feature_restore_indx, get_thresholds
import numpy as np

np.set_printoptions(suppress=1)  # prints floats, no scientific notation
np.set_printoptions(precision=3)  # rounds all array elements to 3rd digit
from ..biodivsim.StateInitializer import *
from ..algorithms.reinforce import RichProtectActionAdaptor, RestoreRichStateAdaptor
from ..agents.policy import PolicyNN, PolicyRestore, get_NN_model_prm, PolicyRestoreUpdateFreq
from concurrent.futures import ProcessPoolExecutor
import collections
from .env_setup import *
from ..biodivsim.BioDivEnv import *

DEBUG = 0

class EVOLUTIONBoltzmannBatchRunner(object):
    def __init__(self, state_adaptor, action_adaptor,
                 degrade_steps=5, seed=0,
                 reward_weights=None,
                 fast_fwd_disturbance_steps=50,
                 deterministic=False):
        """
        state_adaptor = RichStateAdaptor()
        action_adaptor = reinforce.py: RichProtectActionAdaptor(grid_size, RESOLUTION)
        """
        self._state_adaptor = state_adaptor
        self._action_adaptor = action_adaptor
        self._degrade_steps = degrade_steps
        self._fast_fwd_disturbance = fast_fwd_disturbance_steps
        self._rewards = []
        self._seed = seed
        self._deterministic = deterministic
        if reward_weights is None:
            self._reward_weights = {'carbon': 10, 'species_risk': 10}
        else:
            self._reward_weights = reward_weights
        self._rs = get_rnd_gen(seed)


    def select_action(self, state, info, policy):
        # print(np.sum(state['protection_matrix']), "select next action")
        # print("grid_obj_previous", state['grid_obj_previous'].numberOfIndividuals(),
        #       "grid_obj_most_recent", state['grid_obj_most_recent'].numberOfIndividuals())
        state = self._state_adaptor.adapt(state, info)
        probs = policy.probs(state)
        if self._deterministic:
            action = np.argmax(probs)
        else:
            action = self._rs.choice(policy.num_output, 1, p=probs)
        if DEBUG:
            print("Pmax action:", np.argmax(probs), "selected:", action,
                  "Pmax:", np.round(np.max(probs),2), "Pavg:", np.round(np.mean(probs), 2), "Pselected:", probs[action])
        return self._action_adaptor.adapt(action.item())

    def run_episode(self, env, policy, noise):

        del self._rewards[:]
        # print('apply noise')
        policy.perturbeParams(noise)

        # # init env (randomize and re-load pickle file)
        if self._seed:
            r = self._seed
        else:
            r = np.abs(np.rint(1 + 1000 * noise[0])).astype(int)
        list_species_values = init_sp_values(env.bioDivGrid._n_species,
                                             seed=r,
                                             grid_size=env.bioDivGrid.length)
        # env.list_species_values = list_species_values
        env.list_species_values_init = list_species_values
        env.update_previous_observation = True
        env.set_reward_weights(self._reward_weights)
        _ = env.reset(fullInfo=True)
        # PRE EVOLVE ENV
        # add heterogeneity in carrying capacity
        env.heterogeneous_carrying_capacity(baseline_K=0.3,
                                            species_cell_K=False,
                                            seed=r+1)

        # fix 'natural' baseline for feature extraction
        env.observe()
        env.update_previous_observation = False
        if DEBUG:
            print(r, list_species_values[0], np.sum(env.bioDivGrid.h))

        # --- 2. fast-forward disturbance and environment degradation ---#
        distb_obj, _ = get_disturbance(8,seed=r+2)
        dm_tmp = env.bioDivGrid.disturbance_matrix + 0
        for i in range(self._fast_fwd_disturbance):
            dm_tmp = distb_obj.updateDisturbance(dm_tmp)
        env.bioDivGrid.setDisturbanceMatrix(dm_tmp)
        # env.bioDivGrid.setSelectiveDisturbanceMatrix(dm_tmp)
        # env.evolve(self._degrade_steps, verbose=True)  # degrade environment
        env.fast_forward(self._degrade_steps, verbose=False)
        # set current env as starting state
        env.reset_init_values()
        env.observe()
        state = env._enrichObs()
        info = env._getInfo()
        if DEBUG:
            print("NEW h", np.sum(env.bioDivGrid.h))
            print(" env.risk_label_counts(normalize=True)",  env.risk_label_counts(normalize=True))
        ep_reward = 0

        # for t in range(1, env.iterations):  # Don't infinite loop while learning
        while True:
            action = self.select_action(state, info, policy)
            state, reward, done, info = env.step(action)
            # state here is richObs in BioDivEnv which is lastObs.stats_quadrant + timeSince last observe
            self._rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # print(np.array(env.history))
        if DEBUG:
            print(" env.risk_label_counts(normalize=True)",  env.risk_label_counts(normalize=True))
        return self._rewards, info


class EVOLUTIONBoltzmannBatchEmpirical(EVOLUTIONBoltzmannBatchRunner):
    def __init__(self, state_adaptor, action_adaptor,
                 degrade_steps=5, seed=0,
                 reward_weights=None,
                 fast_fwd_disturbance_steps=50,
                 feature_update_per_step=True,
                 actions_per_step=1,
                 deterministic=False):
        """
        state_adaptor = RichStateAdaptor()
        action_adaptor = reinforce.py: RichProtectActionAdaptor(grid_size, RESOLUTION)
        """
        super().__init__(state_adaptor, action_adaptor, degrade_steps, seed,
                 reward_weights, fast_fwd_disturbance_steps, deterministic=deterministic)

        self.feature_update_per_step = feature_update_per_step
        self._actions_per_step = actions_per_step


    def run_episode(self, env, policy, noise):

        del self._rewards[:]
        # print('apply noise')
        policy.perturbeParams(noise)

        # # init env (randomize and re-load pickle file)
        env.set_reward_weights(self._reward_weights)

        state = env._enrichObs()
        info = env._getInfo()
        # print("EVOLUTIONBoltzmannBatchEmpirical run_episode: grid_obj_previous, grid_obj",
        # np.sum(state["grid_obj_previous"].h), np.sum(state["grid_obj_most_recent"].h))

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
            action = self.select_action(state, info, policy)
            state, reward, done, info = env.step(action, skip_env_step=skip_env_step)
            if skip_env_step is False:
                env.species_risk_criteria.update_pop_sizes(env.bioDivGrid)
                # decide if update features or not
                if self.feature_update_per_step:
                    policy.reset_lastObs(None)

            # state here is richObs in BioDivEnv which is lastObs.stats_quadrant + timeSince last observe
            self._rewards.append(reward)
            ep_reward += reward
            counter += 1
            if done:
                break

        # print(np.array(env.history))
        if DEBUG:
            print(" env.risk_label_counts(normalize=True)",  env.risk_label_counts(normalize=True))
        return self._rewards, info


RunnerInput = collections.namedtuple("RunnerInput", ("env", "policy", "runner"))
EvolutionRunnerInput = collections.namedtuple(
    "EvolutionRunnerInput", ("env", "policy", "runner", "noise")
)


def runOneEvolutionEpoch(runnerInput):
    env = runnerInput.env
    policy = runnerInput.policy
    runner = runnerInput.runner
    param_noise = runnerInput.noise
    # print('run episode')
    rewards, info = runner.run_episode(env, policy, param_noise)
    # TODO log / store probs for monitoring
    return info, rewards, []


def computeEvolutionaryUpdate(results,
                              epoch_coeff,
                              noise, alpha,
                              sigma,
                              running_reward):
    if sigma == 0:
        return epoch_coeff
    final_reward_list = []
    for res in results:
        final_reward_list.append(np.sum(res[1]))

    n = len(final_reward_list)
    perturbed_advantage = [
        (rr - running_reward) * nn for rr, nn in zip(final_reward_list, noise)
    ]
    # perturbed_advantage has the size ( batch_size, coeff_size )
    new_coeff = epoch_coeff + alpha / (n * sigma) * np.sum(perturbed_advantage, 0)
    return new_coeff


def getFinalStepAvgReward(results):
    avg_final_rew = 0
    count = 0
    for res in results:
        avg_final_rew += np.sum(res[1])
        count += 1

    if count > 0:
        return avg_final_rew / count
    else:
        return 0


def runBatchGeneticStrategyRichPolicy(batch_size,
                                      epochs,
                                      time_steps,
                                      budget,
                                      lr,
                                      lr_adapt,
                                      temperature=1,
                                      max_workers=0,
                                      outfile="",
                                      disturbance_mode=0,
                                      seed=0,
                                      obsMode=1,
                                      runMode=RunMode.ORACLE,
                                      observe_error=0,
                                      running_reward_start=-1000,
                                      eps_running_reward=0.5,
                                      sigma=1.0,
                                      use_protection_cost=0,
                                      wNN=None,
                                      n_NN_nodes=[4, 0],
                                      increase_temp=0,
                                      rewardMode="species",
                                      random_training=1,
                                      resolution=np.array([1,1]),
                                      dispersal_rate=0.1,
                                      climate_obj=0,
                                      climate_as_disturbance=0,
                                      rnd_alpha_species=0,
                                      disturbance_dep_dispersal=0,
                                      max_fraction_protected=1,
                                      edge_effect=0,
                                      growth_rates=[0.1],
                                      wd="",
                                      max_temperature=10,
                                      sp_threshold_feature_extraction=1,
                                      start_protecting=1,
                                      degrade_steps=5,
                                      fast_fwd_disturbance_steps=50,
                                      reward_weights=None,
                                      wd_output=None
                                      ):
    RESOLUTION = resolution
    if max_workers == 0:
        max_workers = batch_size
    if random_training == 1:
        rnd_disturbance_init = disturbance_mode
        gridInitializer = RandomPickleInitializer(pklfolder=wd, verbose=True)
    elif random_training == 0:
        rnd_disturbance_init = -1
        gridInitializer = PickleInitializerBatch(
            pklfolder=wd, verbose=True, pklfile_i=0
        )
    elif random_training == 2:
        rnd_disturbance_init = -1
        gridInitializer = PickleInitializerSequential(pklfolder=wd, verbose=True)
    init_data = gridInitializer.getInitialState(1, 1, 1)
    n_cells = init_data.shape[1]
    n_species = init_data.shape[0]
    alpha = 0.01
    K_max = np.einsum("xyz -> yz", init_data)[0][0]

    grid_size = n_cells
    OUTPUT = (grid_size ** 2) / (RESOLUTION[0] * RESOLUTION[1])
    if OUTPUT % int(OUTPUT) != 0:
        sys.exit("\n\nResolution not allowed!\n\n")
    else:
        OUTPUT = int(OUTPUT)
        print("Number of protection units: ", OUTPUT)
    
    distb_obj, selectivedistb_obj = get_disturbance(disturbance_mode)
    disturbance_sensitivity = np.zeros(n_species) + np.random.random(n_species)
    selective_sensitivity = np.random.beta(0.2, 0.7, n_species)
    climate_sensitivity = np.random.beta(2, 2, n_species)
    if random_training:
        list_species_values = []
    else:
        (
            disturbance_sensitivity,
            selective_sensitivity,
            climate_sensitivity,
        ) = init_sp_sensitivities(n_species, seed=seed)
        list_species_values = init_sp_values(n_species, seed=seed, grid_size=n_cells)

        dis_sel_cli = [init_sp_sensitivities(n_species, seed=seed+b) for b in range(batch_size)]
        # print(dis_sel_cli)
        # print(len(dis_sel_cli), len(dis_sel_cli[0]))
        list_sp_val_l = [init_sp_values(n_species, seed=seed+b, grid_size=n_cells) for b in range(batch_size)]




    # TODO: expose this
    feature_set = rewardMode #"all" # rewardMode
    num_features = len(get_feature_restore_indx(mode=feature_set))
    print("num_features", num_features)
    print(get_feature_restore_indx(mode=feature_set))
    # quit()
    [
        num_output,
        num_meta_features,
        nodes_layer_1,
        nodes_layer_2,
        nodes_layer_3,
        n_prms,
    ] = get_NN_model_prm(num_features, n_NN_nodes, OUTPUT)

    if wNN is None:
        coeff_features = np.abs(np.random.normal(0, 0.1, n_prms))
        coeff_meta_features = np.random.normal(0, 0.1, num_meta_features)
    else:
        coeff_features = wNN[:-num_meta_features]
        coeff_meta_features = wNN[-num_meta_features:]

    state_adaptor = RestoreRichStateAdaptor()
    action_adaptor = RichProtectActionAdaptor(grid_size, RESOLUTION)
    # init out file
    with open(os.path.join(wd_output, outfile), "w") as f:
        writer = csv.writer(f, delimiter="\t")
        head = [
            "epoch",
            "reward",
            "protected_cells",
            "budget_left",
            "time_last_protect",
            "running_reward",
            "avg_cost",
            "extant_sp",
            "extant_sp_value",
            "extant_sp_pd",
            "tot_carbon"
            ""
        ]

        head = head + ['CR/EX', 'EN', 'VU', 'NT', 'LC']
        if rewardMode == "ext_risk_carbon":
            head = head + ['reward_c', 'reward_sp']
        head = head + ["coeff_%s" % i for i in range(len(coeff_features))]
        head = head + ["threshold_%s" % i for i in range(num_meta_features)]
        writer.writerow(head)
    # TODO end refactor
    evolutionRunner = EVOLUTIONBoltzmannBatchRunner(state_adaptor,
                                                    action_adaptor,
                                                    degrade_steps=degrade_steps,
                                                    seed=seed,
                                                    reward_weights=reward_weights,
                                                    fast_fwd_disturbance_steps=fast_fwd_disturbance_steps
                                                    )

    if random_training:
        envInput = [
            EnvInput(
                budget,
                gridInitializer,
                n_cells,
                n_species,
                alpha,
                K_max,
                dispersal_rate,
                distb_obj,
                disturbance_sensitivity,
                selectivedistb_obj,
                selective_sensitivity,
                climate_obj,
                climate_sensitivity,
                time_steps,
                runMode,
                i,
                obsMode,
                use_protection_cost,
                random_training,
                rnd_disturbance_init,
                rewardMode,
                list_species_values,
                RESOLUTION,
                climate_as_disturbance,
                rnd_alpha_species,
                disturbance_dep_dispersal,
                max_fraction_protected,
                edge_effect,
                growth_rates,
                start_protecting,
            )
            for i in range(batch_size)
        ]
    else:  # random_training = 0 (not random)
        gridInitializer_list = [
            PickleInitializerBatch(pklfolder=wd, verbose=True, pklfile_i=i)
            for i in range(batch_size)
        ]
        envInput = [
            EnvInput(
                budget,
                gridInitializer_list[i],
                n_cells,
                n_species,
                alpha,
                K_max,
                dispersal_rate,
                distb_obj,
                dis_sel_cli[i][0], # disturbance_sensitivity,
                selectivedistb_obj,
                dis_sel_cli[i][1], # selective_sensitivity,
                climate_obj,
                dis_sel_cli[i][2], # climate_sensitivity,
                time_steps,
                runMode,
                i,
                obsMode,
                use_protection_cost,
                random_training,
                rnd_disturbance_init,
                rewardMode,
                list_sp_val_l[i], #list_species_values,
                RESOLUTION,
                climate_as_disturbance,
                rnd_alpha_species,
                disturbance_dep_dispersal,
                max_fraction_protected,
                edge_effect,
                growth_rates,
                start_protecting,
            )
            for i in range(batch_size)
        ]

    print("max_workers", max_workers, batch_size)
    if batch_size > 1:  # parallelize
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            envList = list(pool.map(buildEnv, envInput))
    else:
        envList = [buildEnv(envInput[0])]

    policy = PolicyRestore(num_features,
                           num_meta_features,
                           num_output,
                           coeff_features,
                           coeff_meta_features,
                           temperature,
                           mode=obsMode,
                           feature_set=feature_set,
                           observe_error=observe_error,
                           nodes_l1=nodes_layer_1,
                           nodes_l2=nodes_layer_2,
                           nodes_l3=nodes_layer_3,
                           sp_threshold=sp_threshold_feature_extraction,
                           plot_features=False,
                           quadrant_coords_list=envList[0].quadrant_coords_list,
                           wd_plot=wd_output
                           )



    print("=============================================")
    print("setup done! Running parameter optimization...")
    print("=============================================")

    running_reward = running_reward_start

    for epoch in range(epochs):
        epoch_coeff = policy.coeff
        lr_epoch = np.max([0.05, lr * np.exp(-lr_adapt * epoch)])
        if increase_temp and epoch > 0:
            if policy.temperature < max_temperature:
                policy.setTemperature(policy.temperature + increase_temp)
                print(f"increase temperature to {policy.temperature}; lr = {lr_epoch}")

        print("=======================================")
        print(f"running epoch {epoch}")
        print("=======================================")

        param_noise = (
            np.random.normal(
                0, 1, (batch_size, len(coeff_features) + num_meta_features)
            )
            * sigma
        )
        if batch_size > 1:  # parallelize
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                runnerInputList = [
                    EvolutionRunnerInput(env, policy, evolutionRunner, noise)
                    for env, noise in zip(envList, param_noise)
                ]
                results = list(pool.map(runOneEvolutionEpoch, runnerInputList))
        else:
            runnerInputList = [
                EvolutionRunnerInput(env, policy, evolutionRunner, noise)
                for env, noise in zip(envList, param_noise)
            ]
            results = [runOneEvolutionEpoch(runnerInputList[0])]

        avg_reward = getFinalStepAvgReward(results)
        # moving average of reward
        if epoch == 0 and running_reward_start == -1000:
            running_reward = avg_reward
        running_reward = (
            eps_running_reward * avg_reward
            + (1.0 - eps_running_reward) * running_reward
        )
        newCoeff = computeEvolutionaryUpdate(
            results, epoch_coeff, param_noise, lr_epoch, sigma, running_reward
        )

        policy.setCoeff(newCoeff)

        print("=======================================")
        print(f"epoch {epoch} summary")
        print("=======================================")
        print(f"policy coeff: {policy.coeff}")
        print(f"avg reward: {avg_reward}")
        print("rewards", [np.sum(res[1]) for res in results])
        print("budget left", [res[0]["budget_left"] for res in results])
        print("time last protect", [res[0]["time_last_protect"] for res in results])
        print(
            "n. protected cells", [res[0]["NumberOfProtectedCells"] for res in results]
        )
        print("costs", [res[0]["CostOfProtection"] for res in results])
        avg_budget_left = np.mean([res[0]["budget_left"] for res in results])
        avg_time_last_protect = np.mean(
            [res[0]["time_last_protect"] for res in results]
        )
        avg_protected_cells = np.mean(
            [res[0]["NumberOfProtectedCells"] for res in results]
        )
        avg_cost = np.mean([res[0]["CostOfProtection"] for res in results])
        avg_extant_sp = np.mean([res[0]["ExtantSpecies"] for res in results])
        avg_extant_sp_value = np.mean([res[0]["ExtantSpeciesValue"] for res in results])
        avg_extant_sp_pd = np.mean([res[0]["ExtantSpeciesPD"] for res in results])

        with open(os.path.join(wd_output, outfile), "a") as f:
            writer = csv.writer(f, delimiter="\t")
            l = [
                epoch,
                avg_reward,
                avg_protected_cells,
                avg_budget_left,
                avg_time_last_protect,
                running_reward,
                avg_cost,
                avg_extant_sp,
                avg_extant_sp_value,
                avg_extant_sp_pd,

                np.mean([res[0]['TotalCarbon'] for res in results]),
                np.mean([res[0]['CR/EX'] for res in results]),
                np.mean([res[0]['EN'] for res in results]),
                np.mean([res[0]['VU'] for res in results]),
                np.mean([res[0]['NT'] for res in results]),
                np.mean([res[0]['LC'] for res in results])

            ]

            if rewardMode == "ext_risk_carbon":
                avg_reward_c = np.mean([res[0]['reward_c'] for res in results])
                avg_reward_sp = np.mean([res[0]['reward_sp'] for res in results])
                l = l + [avg_reward_c, avg_reward_sp]

            l = l + list(policy.coeff)
            writer.writerow(l)


def train_restore_model(rnd_seed=1234,
                        # TODO: fix obsMode options
                        obsMode=0,  # 0: random, 1: full monitor, 2: citizen-science, 3: one-time, 4: value, 5: area
                        batchSize=3,
                        steps=10,
                        epochs=100,
                        observePolicy=1,  #  0: NO-OBSERVE-UPDATE 1: ORACLE 2: PROTECTATONCE
                        disturbance=4,
                        protection_cost=1,
                        n_nodes=[4, 0],
                        random_training=1,  # "0: fixed training; 1: random; 2: fixed training, seq pickle"
                        rewardMode="species",
                        obs_error=0,  # "Amount of error in species counts (feature extraction)"
                        resolution=np.array([5, 5]),
                        budget=55,
                        max_fraction_protected=1,
                        dispersal_rate=0.1,
                        use_climate=0,  # "0: no climate change, 1: climate change, 2: climate disturbance, 3: climate change + random variation"
                        climate_change_magnitude=0.1,
                        peak_anomaly=2,
                        rnd_alpha=0,  # (st.dev of species specific fluctuation in mortality (if 'by_species' ==1 in SimpleGrid)
                        dist_dependent_dispersal=0,
                        outfile="training_output.log",
                        # training settings
                        sigma=1,
                        temperature=1,
                        increase_temp=1 / 100,  # temperature = 10 after 1000 epochs
                        lr=0.5,
                        lr_adapt=0.01,
                        wNN=None,
                        running_reward_start=-1000,  # i.e. re-initialized at epoch 0,
                        eps_running_reward=0.25,  # if eps=1 running_reward = last reward
                        wd="data_dependencies/pickles",
                        grid_size=50,
                        growth_rates=[0.1],
                        edge_effect=0,
                        max_temperature=10,
                        sp_threshold_feature_extraction=1,
                        start_protecting=1,
                        degrade_steps=5,
                        fast_fwd_disturbance_steps=50,
                        reward_weights=None,
                        wd_output=""
                        ):

    """
        sp_threshold = 10

    grid_size = 50
    growth_rate = [0.1]
    edge_effect = 0
    max_temperature = 10"""

    runMode = [RunMode.NOUPDATEOBS, RunMode.ORACLE, RunMode.PROTECTATONCE][
        observePolicy
    ]
    climate_disturbance = 0
    if use_climate == 1:
        climate_change = climate_change_magnitude
        from ..biodivsim.ClimateGenerator import (
            SimpleGradientClimateGenerator as ClimateGen,
        )

        CLIMATE_OBJ = ClimateGen(0, climate_change=climate_change)
    elif use_climate == 2:
        climate_disturbance = 1
        from ..biodivsim.ClimateGenerator import RegionalClimateGenerator as ClimateGen

        CLIMATE_OBJ = ClimateGen(0)
    elif use_climate == 3:  # global warming + random variation
        climate_change = climate_change_magnitude
        PEAK_ANOMALY = peak_anomaly
        from ..biodivsim.ClimateGenerator import (
            GradientClimateGeneratorRnd as ClimateGen,
        )

        CLIMATE_OBJ = ClimateGen(
            0, climate_change=climate_change, peak_anomaly=PEAK_ANOMALY
        )
    else:
        CLIMATE_OBJ = 0

    # get_feature_restore_indx(obsMode, print_obs_mode=True)

    runBatchGeneticStrategyRichPolicy(batch_size=batchSize,
                                      epochs=epochs,
                                      time_steps=steps,
                                      budget=budget,
                                      lr=lr,
                                      lr_adapt=lr_adapt,
                                      temperature=temperature,
                                      outfile=outfile,
                                      disturbance_mode=disturbance,
                                      seed=rnd_seed,
                                      obsMode=obsMode,
                                      runMode=runMode,
                                      observe_error=obs_error,
                                      running_reward_start=running_reward_start,
                                      eps_running_reward=eps_running_reward,
                                      sigma=sigma,
                                      use_protection_cost=protection_cost,
                                      rewardMode=rewardMode,
                                      wNN=wNN,
                                      n_NN_nodes=n_nodes,
                                      increase_temp=increase_temp,
                                      random_training=random_training,
                                      resolution=resolution,
                                      dispersal_rate=dispersal_rate,
                                      climate_obj=CLIMATE_OBJ,
                                      climate_as_disturbance=climate_disturbance,
                                      rnd_alpha_species=rnd_alpha,
                                      disturbance_dep_dispersal=dist_dependent_dispersal,
                                      max_fraction_protected=max_fraction_protected,
                                      edge_effect=edge_effect,
                                      growth_rates=growth_rates,
                                      wd=wd,
                                      max_temperature=max_temperature,
                                      sp_threshold_feature_extraction=sp_threshold_feature_extraction,
                                      start_protecting=start_protecting,
                                      degrade_steps=degrade_steps,
                                      fast_fwd_disturbance_steps=fast_fwd_disturbance_steps,
                                      reward_weights=reward_weights,
                                      wd_output=wd_output
                                      )



############ EMPIRICAL TRAINING
def runBatchGeneticStrategyEmpirical(envList,
                                     epochs,
                                     lr,
                                     lr_adapt,
                                     temperature=1,
                                     max_workers=0,
                                     outfile="",
                                     obsMode=1,
                                     observe_error=0,
                                     running_reward_start=-1000,
                                     eps_running_reward=0.5,
                                     sigma=1.0,
                                     wNN=None,
                                     n_NN_nodes=[4, 0],
                                     increase_temp=0,
                                     resolution=np.array([1, 1]),
                                     max_temperature=10,
                                     sp_threshold_feature_extraction=1,
                                     wd_output=None,
                                     actions_per_step=1,
                                     return_env=False,
                                     deterministic=False
                                     ):
    batch_size = len(envList)
    RESOLUTION = resolution
    grid_size = envList[0].bioDivGrid.h.shape[1]
    if max_workers == 0:
        max_workers = batch_size

    OUTPUT = envList[0].num_quadrants
    OUTPUT = int(OUTPUT)
    print("Number of protection units: ", OUTPUT)

    # TODO: expose this
    rewardMode = envList[0].rewardMode
    feature_set = rewardMode  # "all" # rewardMode
    num_features = len(get_feature_restore_indx(mode=feature_set))
    print("num_features", num_features)
    print(get_feature_restore_indx(mode=feature_set))
    # quit()
    [
        num_output,
        num_meta_features,
        nodes_layer_1,
        nodes_layer_2,
        nodes_layer_3,
        n_prms,
    ] = get_NN_model_prm(num_features, n_NN_nodes, OUTPUT)

    if wNN is None:
        coeff_features = np.abs(np.random.normal(0, 0.1, n_prms))
        coeff_meta_features = np.random.normal(0, 0.1, num_meta_features)
    else:
        coeff_features = wNN
        coeff_meta_features = np.array([])

    state_adaptor = RestoreRichStateAdaptor()
    action_adaptor = RichProtectActionAdaptor(grid_size, RESOLUTION)
    # init out file
    with open(os.path.join(wd_output, outfile), "w") as f:
        writer = csv.writer(f, delimiter="\t")
        head = [
            "epoch",
            "reward",
            "protected_cells",
            "budget_left",
            "time_last_protect",
            "running_reward",
            "avg_cost",
            "extant_sp",
            "extant_sp_value",
            "extant_sp_pd",
            "tot_carbon"
            ""
        ]

        head = head + ['CR/EX', 'EN', 'VU', 'NT', 'LC']
        if rewardMode == "ext_risk_carbon":
            head = head + ['reward_c', 'reward_sp']
        head = head + ["coeff_%s" % i for i in range(len(coeff_features))]
        head = head + ["threshold_%s" % i for i in range(num_meta_features)]
        writer.writerow(head)
    # TODO end refactor
    evolutionRunner = EVOLUTIONBoltzmannBatchEmpirical(state_adaptor, action_adaptor,
                                                       actions_per_step=actions_per_step,
                                                       deterministic=deterministic)

    policy = PolicyRestoreUpdateFreq(
        num_features,
        num_meta_features,
        num_output,
        coeff_features,
        coeff_meta_features,
        temperature,
        mode=obsMode,
        feature_set=feature_set,
        observe_error=observe_error,
        use_true_natural_state=True,
        nodes_l1=nodes_layer_1,
        nodes_l2=nodes_layer_2,
        nodes_l3=nodes_layer_3,
        sp_threshold=sp_threshold_feature_extraction,
        quadrant_coords_list=envList[0].quadrant_coords_list,
        wd_plot=wd_output
    )

    print("policy._coeff", policy._coeff, policy.coeff)

    print("=============================================")
    print("setup done! Running parameter optimization...")
    print("=============================================")

    running_reward = running_reward_start

    for epoch in range(epochs):
        # re-init env
        env_list_init = copy.deepcopy(envList)
        #TODO "can replace w generator to randomize sensitivities etc"
        #
        epoch_coeff = policy.coeff
        lr_epoch = np.max([0.05, lr * np.exp(-lr_adapt * epoch)])
        if increase_temp and epoch > 0:
            if policy.temperature < max_temperature:
                policy.setTemperature(policy.temperature + increase_temp)
                print(f"increase temperature to {policy.temperature}; lr = {lr_epoch}")

        print("=======================================")
        print(f"running epoch {epoch}")
        print("=======================================")

        param_noise = (
                np.random.normal(
                    0, 1, (batch_size, len(coeff_features) + num_meta_features)
                )
                * sigma
        )
        if batch_size > 1:  # parallelize
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                runnerInputList = [
                    EvolutionRunnerInput(env, policy, evolutionRunner, noise)
                    for env, noise in zip(env_list_init, param_noise)
                ]
                results = list(pool.map(runOneEvolutionEpoch, runnerInputList))
        else:
            runnerInputList = [
                EvolutionRunnerInput(env, policy, evolutionRunner, noise)
                for env, noise in zip(env_list_init, param_noise)
            ]
            results = [runOneEvolutionEpoch(runnerInputList[0])]

        avg_reward = getFinalStepAvgReward(results)
        # moving average of reward
        if epoch == 0 and running_reward_start == -1000:
            running_reward = avg_reward
        running_reward = (
                eps_running_reward * avg_reward
                + (1.0 - eps_running_reward) * running_reward
        )
        newCoeff = computeEvolutionaryUpdate(
            results, epoch_coeff, param_noise, lr_epoch, sigma, running_reward
        )

        policy.setCoeff(newCoeff)

        print("=======================================")
        print(f"epoch {epoch} summary")
        print("=======================================")
        print(f"policy coeff: {policy.coeff}")
        print(f"avg reward: {avg_reward}")
        print("rewards", [np.sum(res[1]) for res in results])
        print("budget left", [res[0]["budget_left"] for res in results])
        print("time last protect", [res[0]["time_last_protect"] for res in results])
        print(
            "n. protected cells", [res[0]["NumberOfProtectedCells"] for res in results]
        )
        print("costs", [res[0]["CostOfProtection"] for res in results])
        avg_budget_left = np.mean([res[0]["budget_left"] for res in results])
        avg_time_last_protect = np.mean(
            [res[0]["time_last_protect"] for res in results]
        )
        avg_protected_cells = np.mean(
            [res[0]["NumberOfProtectedCells"] for res in results]
        )
        avg_cost = np.mean([res[0]["CostOfProtection"] for res in results])
        avg_extant_sp = np.mean([res[0]["ExtantSpecies"] for res in results])
        avg_extant_sp_value = np.mean([res[0]["ExtantSpeciesValue"] for res in results])
        avg_extant_sp_pd = np.mean([res[0]["ExtantSpeciesPD"] for res in results])

        with open(os.path.join(wd_output, outfile), "a") as f:
            writer = csv.writer(f, delimiter="\t")
            l = [
                epoch,
                avg_reward,
                avg_protected_cells,
                avg_budget_left,
                avg_time_last_protect,
                running_reward,
                avg_cost,
                avg_extant_sp,
                avg_extant_sp_value,
                avg_extant_sp_pd,

                np.mean([res[0]['TotalCarbon'] for res in results]),
                np.mean([res[0]['CR/EX'] for res in results]),
                np.mean([res[0]['EN'] for res in results]),
                np.mean([res[0]['VU'] for res in results]),
                np.mean([res[0]['NT'] for res in results]),
                np.mean([res[0]['LC'] for res in results])

            ]

            if rewardMode == "ext_risk_carbon":
                avg_reward_c = np.mean([res[0]['reward_c'] for res in results])
                avg_reward_sp = np.mean([res[0]['reward_sp'] for res in results])
                l = l + [avg_reward_c, avg_reward_sp]

            l = l + list(policy.coeff)
            writer.writerow(l)

    if return_env:
        return runnerInputList