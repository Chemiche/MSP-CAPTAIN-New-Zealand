import numpy as np
import sys
from ..biodivsim.SimGrid import SimGrid

class ExtinctionRisk():
    def __init__(self,
                 natural_state: SimGrid,
                 relative_range_thresholds=None,
                 # risk_weights=np.array([-3, -2, -1, 0, 1])
                 risk_weights=np.array([-12, -3, -1, 0, 1]),
                 evolve_status=None,
                 starting_rl_status=None,
                 current_state: SimGrid = None,
                 upgrade_protected_species=False,
                 min_individuals_cell=None,
                 ):
        if relative_range_thresholds is None:
            relative_range_thresholds = np.array([0.05, # CR / EX
                                                  0.10, # EN
                                                  0.20, # VU
                                                  0.50 # NT
                                                    # LC
                                                  ])

        if current_state is None:
            current_state = natural_state


        self.relative_range_thresholds = relative_range_thresholds
        self.natural_sp_range = natural_state.geoRangePerSpecies()
        self.current_sp_range = current_state.geoRangePerSpecies()
        self.available_labels = np.arange(5)
        self.available_labels_text = ['CR/EX', 'EN', 'VU', 'NT', 'LC']
        self.n_labels = len(self.available_labels)
        self.risk_weights = risk_weights
        self._use_min_area = False
        self._init_pop_size = None
        self.starting_rl_status = starting_rl_status
        self.evolve_status = evolve_status
        self._upgrade_protected_species = upgrade_protected_species
        self._min_individuals_cell = min_individuals_cell

    def reset_thresholds(self, t):
        self.relative_range_thresholds = t

    def set_min_area(self, min_a=None):
        self._use_min_area = True
        if min_a is None:
            min_a = np.array([1, 2, 5])

        self._min_area = min_a

    def set_init_pop_sizes(self, pop):
        self._init_pop_size = pop


    def classify_species(self, state: SimGrid):
        range_fraction = state.geoRangePerSpecies() / (self.natural_sp_range + 0.1)
        labels = np.digitize(range_fraction,
                             bins=self.relative_range_thresholds)

        if self._use_min_area:
            labels[state.geoRangePerSpecies() <= self._min_area[2]] = 2
            labels[state.geoRangePerSpecies() <= self._min_area[1]] = 1
            labels[state.geoRangePerSpecies() <= self._min_area[0]] = 0

        if self._upgrade_protected_species:
            protected_range_fraction = state.protectedRangePerSpecies()
            labels[protected_range_fraction > 0] = [protected_range_fraction > 0] + 1
            labels[labels >= len(self.available_labels)] = len(self.available_labels)


        return labels

    def update_pop_sizes(self):
        pass


class ExtinctionRiskProtectedRange(ExtinctionRisk):
    def __init__(self,
                 natural_state: SimGrid,
                 relative_range_thresholds=None,
                 risk_weights = np.array([-12, -3, -1, 0, 1]),
                 current_state: SimGrid = None,
                 evolve_status=None,
                 starting_rl_status = None,
                 protect_fraction_upgrade_thresholds= None,
                 min_range_change=0.05
                 ):
        super().__init__(natural_state,
                         relative_range_thresholds,
                         risk_weights,
                         current_state=current_state,
                         evolve_status=evolve_status,
                         starting_rl_status=starting_rl_status)

        if protect_fraction_upgrade_thresholds is None:
            protect_fraction_upgrade_thresholds = np.array([0.1, 0.25, 0.5]) # -> add +1, +2, +3
        self.protect_fraction_upgrade_thresholds = protect_fraction_upgrade_thresholds

        self.previous_range_fraction = None
        self.previous_labels = None
        self.min_range_change = min_range_change

        # if self.starting_rl_status is None:
        #     self.starting_rl_status = self.classify_species(natural_state)


    def classify_species(self, state: SimGrid):
        # TODO: apply change only if >5% from previous?
        # classify based on current vs natura range size
        range_fraction = state.geoRangePerSpecies() / (self.natural_sp_range + 0.001)
        # env_optim.bioDivGrid.geoRangePerSpecies() / env_optim.grid_obj_previous.geoRangePerSpecies()
        labels = np.digitize(np.round(range_fraction * 100),
                             bins=100 * self.relative_range_thresholds)

        if self.previous_range_fraction is not None:
            print(self.previous_range_fraction[:5], "range_fraction", range_fraction[:5])
            indx = np.abs(self.previous_range_fraction - range_fraction) < self.min_range_change
            labels[indx] = self.previous_labels[indx]

        # adjust based on fraction of protected range out of the current range
        protected_range_fraction = state.protectedRangePerSpecies() / (self.natural_sp_range + 0.001)

        labels_upgraded = labels + np.digitize(protected_range_fraction,
                                               bins=self.protect_fraction_upgrade_thresholds)

        labels_upgraded[labels_upgraded >= len(self.available_labels)] = len(self.available_labels)

        # set starting RL values on first call
        # if self.starting_rl_status is None:
        #     self.starting_rl_status = labels_upgraded

        # self.previous_range_fraction = range_fraction
        # self.previous_labels = labels_upgraded

        return labels_upgraded


class ExtinctionRiskPopSize(ExtinctionRisk):
    def __init__(self,
                 natural_state: SimGrid,
                 relative_range_thresholds=None,
                 risk_weights = np.array([-12, -3, -1, 0, 1]),
                 current_state: SimGrid = None,
                 epsilon = 0.5,
                 evolve_status=None,
                 starting_rl_status=None,
                 # if eps=1 running_reward = last change
                 # eps = 0.5 rolling average
                 # eps < 0.5 longer legacy of long-term change
                 ):
        super().__init__(natural_state,
                         relative_range_thresholds,
                         risk_weights,
                         evolve_status=evolve_status,
                         starting_rl_status=starting_rl_status)
        if current_state is None:
            current_state = natural_state
        self.current_protected_sp_range = current_state.protectedRangePerSpecies()
        self.current_protected_pop_sizes = current_state.protectedIndPerSpecies()
        self.current_sp_range = current_state.geoRangePerSpecies()
        self.current_pop_sizes = current_state.individualsPerSpecies()
        # self.species_status = np.ones(natural_state._n_species) * self.n_labels
        if self.n_labels != 5:
            sys.exit("")
        self.epsilon = epsilon
        self.population_change = np.zeros(natural_state._n_species)

    def classify_species(self, state: SimGrid):
        # TODO: could change the frequency of re-assessment as a monitoring policy
        # species_status_tmp = self.species_status + 0
        current_pop_sizes = state.individualsPerSpecies()
        current_sp_range = state.geoRangePerSpecies()

        range_change = current_sp_range / self.natural_sp_range
        population_change = current_pop_sizes / self.current_pop_sizes - 1
        self.population_change = (self.epsilon * population_change) + (1 - self.epsilon) * self.population_change # rolling avg
        decreasing_pop = current_pop_sizes / self.current_pop_sizes < 0.95

        """
        should check relative change, ie not from natural one but from previous one
        store internally to object when running classify_species function
        """

        range_fraction = state.protectedRangePerSpecies() / self.natural_sp_range
        range_fraction[state.geoRangePerSpecies() > 0.3 * np.max(self.natural_sp_range)] = 0.49 # > 30% -> NT
        range_fraction[state.geoRangePerSpecies() > 0.5 * np.max(self.natural_sp_range)] = 0.99 # > 50% -> LC

        labels = np.digitize(range_fraction,
                             bins=self.relative_range_thresholds)
        # downgrade species w decreasing pop size
        labels[labels > 0] = labels[labels > 0] - (self.population_change[labels > 0] < -0.05).astype(int)
        # upgrade species w increasing pop size
        labels[labels < 4] = labels[labels < 4] + (self.population_change[labels < 4] > 0.0).astype(int)

        self.current_pop_sizes = current_pop_sizes + 0
        self.current_sp_range = current_sp_range + 0

        return labels


class ExtinctioRiskRedList(ExtinctionRisk):
    # DOI: 10.1371/journal.pbio.0020383
    def __init__(self,
                 natural_state: SimGrid,
                 relative_range_thresholds=None,
                 relative_pop_thresholds=np.array([0.10,  # CR / EX
                                                   0.30,  # EN
                                                   0.50,  # VU
                                                   0.70  # NT
                                                   # LC
                                                   ]),
                 reduction_in_population_size=np.array([0.20,  # CR / EX
                                                        0.50,  # EN
                                                        0.70,  # VU
                                                        0.90  # NT
                                                        # LC
                                                        ]),
                 risk_weights = np.array([-12, -3, -1, 0, 1]),
                 current_state: SimGrid = None,
                 epsilon = 0.5,
                 # if eps=1 running_reward = last change
                 # eps = 0.5 rolling average
                 # eps < 0.5 longer legacy of long-term change
                 sufficient_protection = 0.5,
                 evolve_status=None,
                 ):
        super().__init__(natural_state,
                         relative_range_thresholds,
                         risk_weights,
                         evolve_status=evolve_status)
        if current_state is None:
            current_state = natural_state

        self._natural_pop_size = natural_state.individualsPerSpecies()
        self._natural_range_size = natural_state.geoRangePerSpecies()
        self.relative_pop_thresholds = relative_pop_thresholds
        self.reduction_in_population_size_threshold = reduction_in_population_size
        self.current_protected_sp_range = current_state.protectedRangePerSpecies()
        self.current_protected_pop_sizes = current_state.protectedIndPerSpecies()
        self.current_sp_range = current_state.geoRangePerSpecies()
        self.current_pop_sizes = current_state.individualsPerSpecies()
        # self.species_status = np.ones(natural_state._n_species) * self.n_labels
        if self.n_labels != 5:
            sys.exit("")
        self.epsilon = epsilon
        self.sufficient_protection = sufficient_protection
        self.population_change = np.zeros(natural_state._n_species)
        # criterion A
        # if

        # B1, assuming 1 cell = 50x50 km (0.5 degrees)
        """
        if range_size < 1 -> CR
        if range_size < 2 -> EN
        
        
        """
    def classify_species(self, state: SimGrid):
        # TODO: could change the frequency of re-assessment as a monitoring policy
        # species_status_tmp = self.species_status + 0
        current_pop_sizes = state.individualsPerSpecies()
        current_sp_range = state.geoRangePerSpecies()

        population_change = current_pop_sizes / self.current_pop_sizes - 1
        self.population_change = (self.epsilon * population_change) + (
                    1 - self.epsilon) * self.population_change  # rolling avg

        """
        should check relative change, ie not from natural one but from previous one
        store internally to object when running classify_species function
        """

        sp_decreasing_pop = np.where(self.population_change < -0.001)[0]

        range_ratio_natural = current_pop_sizes / self._natural_pop_size

        labelsA1 = np.digitize(range_ratio_natural,
                             bins=self.relative_pop_thresholds)


        labelsA24 = np.digitize(range_ratio_natural,
                             bins=self.reduction_in_population_size_threshold)

        labelsA = labelsA1 + 0
        # species with still decreasing pops have higher thresholds:
        labelsA[sp_decreasing_pop] = labelsA24[sp_decreasing_pop]


        # check fraction of protected range (avoiding 0-divisions)
        range_fraction = state.protectedRangePerSpecies() / (state.geoRangePerSpecies() + 0.1)
        well_protected = np.where(range_fraction > self.sufficient_protection)

        labelsA[well_protected] += 1
        labelsA[labelsA > 4] = 4

        self.current_pop_sizes = current_pop_sizes + 0
        self.current_sp_range = current_sp_range + 0

        return labelsA


class ExtinctioRiskRedListEmpirical(ExtinctionRisk):
    # DOI: 10.1371/journal.pbio.0020383
    def __init__(self,
                 natural_state: SimGrid,
                 relative_range_thresholds=None,
                 risk_weights=np.array([-12, -3, -1, 0, 1]),
                 current_state: SimGrid = None,
                 relative_pop_thresholds=np.array([0.10,  # CR / EX: 0
                                                   0.30,  # EN: 1
                                                   0.50,  # VU: 2
                                                   0.60  # NT: 3
                                                   # LC: 4
                                                   ]),
                 epsilon=0.5,
                 # if eps=1 running_reward = last change
                 # eps = 0.5 rolling average
                 # eps < 0.5 longer legacy of long-term change
                 sufficient_protection=0.5,
                 starting_rl_status=None,
                 pop_decrease_threshold=0.01,
                 evolve_status=True,
                 min_individuals_cell=None,
                 compare_to_init_state=False,
                 ):
        super().__init__(natural_state,
                         relative_range_thresholds,
                         risk_weights,
                         evolve_status=evolve_status,
                         starting_rl_status=starting_rl_status,
                         min_individuals_cell=min_individuals_cell)
        if current_state is None:
            current_state = natural_state

        self._natural_pop_size = natural_state.individualsPerSpecies(self._min_individuals_cell)
        self._natural_range_size = natural_state.geoRangePerSpecies()
        self.current_protected_sp_range = current_state.protectedRangePerSpecies()
        self.current_protected_pop_sizes = current_state.protectedIndPerSpecies(self._min_individuals_cell)
        self.current_sp_range = current_state.geoRangePerSpecies()
        self.current_pop_sizes = current_state.individualsPerSpecies(self._min_individuals_cell)
        self._init_pop_size = current_state.individualsPerSpecies(self._min_individuals_cell)
        # self.species_status = np.ones(natural_state._n_species) * self.n_labels
        if self.n_labels != 5:
            sys.exit("")
        self.epsilon = epsilon
        self.sufficient_protection = sufficient_protection
        self.population_change = np.zeros(natural_state._n_species)
        self.population_change_to_init = np.zeros(natural_state._n_species)
        self._pop_change_threshold = pop_decrease_threshold
        self.relative_pop_thresholds = relative_pop_thresholds
        self._counter = 0
        self.compare_to_init_state = compare_to_init_state
        # criterion A
        # if

        # B1, assuming 1 cell = 50x50 km (0.5 degrees)
        """
        if range_size < 1 -> CR
        if range_size < 2 -> EN


        """
        if starting_rl_status is None:
            self.get_starting_rl_status()
        else:
            self.current_rl_status = starting_rl_status + 0

    def get_starting_rl_status(self):
            print("Initializing RL status based on pop sizes")
            pop_ratio = self.current_pop_sizes / self._natural_pop_size
            self.current_rl_status = np.digitize(pop_ratio,
                                  bins=self.relative_pop_thresholds)
            print(self.current_rl_status)



    def update_pop_sizes(self, state: SimGrid):
        current_pop_sizes = state.individualsPerSpecies(self._min_individuals_cell)
        current_sp_range = state.geoRangePerSpecies()
        # current_protected_pop_sizes = state.protectedIndPerSpecies(self._min_individuals_cell) / current_pop_sizes
        """compare to initial situation"""
        # population_change_to_init = current_pop_sizes / self._init_pop_size - 1

        population_change = current_pop_sizes / self.current_pop_sizes - 1
        self.population_change = (self.epsilon * population_change) + (
                1 - self.epsilon) * self.population_change  # rolling avg

        self._counter = state._counter + 0
        self.current_pop_sizes = current_pop_sizes + 0
        self.current_sp_range = current_sp_range + 0
        rl = self.classify_species(state)
        print("Updating population sizes", rl)
        self.current_rl_status = rl + 0


    def classify_species(self, state: SimGrid):
        rl = self.current_rl_status + 0
        # update prms
        current_pop_sizes = state.individualsPerSpecies(self._min_individuals_cell)
        current_sp_range = state.geoRangePerSpecies()
        current_protected_pop_sizes = state.protectedIndPerSpecies(self._min_individuals_cell) / current_pop_sizes
        """compare to initial situation"""
        population_change_to_init = current_pop_sizes / self._init_pop_size - 1


        # only update if grid moved one step (so multiple calls without steps return
        # same rl status
        if self.evolve_status:
            range_ratio_natural = current_pop_sizes / self._natural_pop_size

            # rl[np.where((self.population_change < -self._pop_change_threshold) & rl == 0)] = 0
            # downgrade
            rl[np.where((self.population_change < -self._pop_change_threshold) # if >CR and decreasing -> CR
                        & (self.current_rl_status >= 1))] -= 1

            # downgrade if pop below threshold relative to starting pop
            tmp_rl = np.digitize(population_change_to_init + 1, self.relative_pop_thresholds)
            rl[tmp_rl < rl] = tmp_rl[tmp_rl < rl]

            # upgrades
            rl[np.where((self.current_rl_status == 0) # if CR but increasing go to EN
                        & (self.population_change > self._pop_change_threshold))] = 1

            rl[np.where((self.current_rl_status >= 1) # if EN but above init pop and increasing -> upgrade
                        & (population_change_to_init >= 0)
                        & (self.population_change > self._pop_change_threshold))] += 1

            # species above 70% pop size
            # rl[np.where((range_ratio_natural > self.relative_pop_thresholds[-1])
            #             & (self.population_change >= -self._pop_change_threshold))] = 4

            # species above 70% pop size and well protected
            # TODO: change to or statement?
            rl[np.where((range_ratio_natural > self.relative_pop_thresholds[-1])
                        & (current_protected_pop_sizes > self.sufficient_protection))] = 4

            #
            # # check fraction of protected range (avoiding 0-divisions)
            # range_fraction = state.protectedRangePerSpecies() / (state.geoRangePerSpecies() + 0.1)
            # well_protected = np.where(range_fraction > self.sufficient_protection)
            #
            # labelsA[well_protected] += 1
            rl[rl > 4] = 4

        return rl





def rnd_init_rl_status(sp_range_size, sp_sensitivity, relative_pop_thresholds=np.array([0.10,  # CR / EX
                                                   0.30,  # EN
                                                   0.50,  # VU
                                                   0.70  # NT
                                                   # LC
                                                   ])):
    sp_range_size_norm = np.log(sp_range_size + 1) / np.mean(np.log(sp_range_size + 1))
    rel_pr = sp_sensitivity * (1 / sp_range_size_norm)
    rel_pr -= np.min(rel_pr)
    pr01 = rel_pr / np.max(rel_pr)
    return np.digitize(pr01, relative_pop_thresholds)







#####################

class ExtinctioRiskCompareNatural(ExtinctionRisk):
    # DOI: 10.1371/journal.pbio.0020383
    def __init__(self,
                 natural_state: SimGrid,
                 relative_range_thresholds=None,
                 risk_weights=np.array([-12, -3, -1, 0, 1]),
                 current_state: SimGrid = None,
                 relative_pop_thresholds=np.array([0.10,  # CR / EX: 0
                                                   0.30,  # EN: 1
                                                   0.50,  # VU: 2
                                                   0.60  # NT: 3
                                                   # LC: 4
                                                   ]),
                 epsilon=0.5,
                 # if eps=1 running_reward = last change
                 # eps = 0.5 rolling average
                 # eps < 0.5 longer legacy of long-term change
                 sufficient_protection=0.5,
                 starting_rl_status=None,
                 pop_decrease_threshold=0.01,
                 evolve_status=True,
                 min_individuals_cell=None,
                 compare_to_init_state=False,
                 ):
        super().__init__(natural_state,
                         relative_range_thresholds,
                         risk_weights,
                         evolve_status=evolve_status,
                         starting_rl_status=starting_rl_status,
                         min_individuals_cell=min_individuals_cell)
        if current_state is None:
            current_state = natural_state

        self._natural_pop_size = natural_state.individualsPerSpecies(self._min_individuals_cell)
        self._natural_range_size = natural_state.geoRangePerSpecies()
        self.current_protected_sp_range = current_state.protectedRangePerSpecies()
        self.current_protected_pop_sizes = current_state.protectedIndPerSpecies(self._min_individuals_cell)
        self.current_sp_range = current_state.geoRangePerSpecies()
        self.current_pop_sizes = current_state.individualsPerSpecies(self._min_individuals_cell)
        self._init_pop_size = current_state.individualsPerSpecies(self._min_individuals_cell)
        # self.species_status = np.ones(natural_state._n_species) * self.n_labels
        if self.n_labels != 5:
            sys.exit("")
        self.epsilon = epsilon
        self.sufficient_protection = sufficient_protection
        self.population_change = np.zeros(natural_state._n_species)
        self.population_change_to_init = np.zeros(natural_state._n_species)
        self._pop_change_threshold = pop_decrease_threshold
        self.relative_pop_thresholds = relative_pop_thresholds
        self._counter = 0
        self.compare_to_init_state = compare_to_init_state
        # criterion A
        # if

        # B1, assuming 1 cell = 50x50 km (0.5 degrees)
        """
        if range_size < 1 -> CR
        if range_size < 2 -> EN


        """
        if starting_rl_status is None:
            self.get_starting_rl_status()
        else:
            self.current_rl_status = starting_rl_status + 0

    def get_starting_rl_status(self):
            print("Initializing RL status based on pop sizes")
            pop_ratio = self.current_pop_sizes / self._natural_pop_size
            self.current_rl_status = np.digitize(pop_ratio,
                                  bins=self.relative_pop_thresholds)
            print(self.current_rl_status)



    def update_pop_sizes(self, state: SimGrid):
        # current_pop_sizes = state.individualsPerSpecies(self._min_individuals_cell)
        # current_sp_range = state.geoRangePerSpecies()
        # # current_protected_pop_sizes = state.protectedIndPerSpecies(self._min_individuals_cell) / current_pop_sizes
        # """compare to initial situation"""
        # # population_change_to_init = current_pop_sizes / self._init_pop_size - 1
        #
        # population_change = current_pop_sizes / self.current_pop_sizes - 1
        # self.population_change = (self.epsilon * population_change) + (
        #         1 - self.epsilon) * self.population_change  # rolling avg
        #
        # self._counter = state._counter + 0
        # self.current_pop_sizes = current_pop_sizes + 0
        # self.current_sp_range = current_sp_range + 0
        # print("Updating population START")
        rl = self.classify_species(state)
        # print("Updating pop sizes    ", rl, "END")
        # self.current_rl_status = rl + 0


    def classify_species(self, state: SimGrid):
        rl = self.current_rl_status + 0
        # update prms
        current_pop_sizes = state.individualsPerSpecies(self._min_individuals_cell)
        current_sp_range = state.geoRangePerSpecies()
        current_protected_pop_sizes = state.protectedIndPerSpecies(self._min_individuals_cell) / current_pop_sizes
        """compare to initial situation"""
        population_change_to_init = current_pop_sizes / self._init_pop_size - 1
        population_change_to_natural = current_pop_sizes / self._natural_pop_size - 1
        # print("Before", rl)
        # print("population_change_to_natural", population_change_to_natural)
        # print("population_change_to_init", population_change_to_init)

        # only update if grid moved one step (so multiple calls without steps return
        # same rl status
        if self.evolve_status:
            pop_ratio_natural = current_pop_sizes / self._natural_pop_size
            rl = np.digitize(pop_ratio_natural, bins=self.relative_pop_thresholds)
            # print("self.current_rl_status", self.current_rl_status)

            # downgrade if pop size decreased compared to initial pop
            indx = np.where(
                (population_change_to_init < -self._pop_change_threshold) &
                (population_change_to_natural < -self._pop_change_threshold)
            )[0]
            # print("indx", indx, self._pop_change_threshold, population_change_to_init[indx], population_change_to_natural[indx])
            rl[indx] = rl[indx] - 1
            rl[rl < 0] = 0

            # upgrade is increased pop size compared to init
            indx = np.where(population_change_to_init > self._pop_change_threshold)[0]
            # print("indx + ", indx, self._pop_change_threshold, population_change_to_init[indx])
            rl[indx] = rl[indx] + 1
            rl[rl > 4] = 4
            # print("new rl                ",    rl, current_pop_sizes[7], self._natural_pop_size[7])
            self.current_rl_status = rl + 0

        # print("After:", rl)

        return rl






class ExtinctioRiskProtectedRange(ExtinctionRisk):
    # DOI: 10.1371/journal.pbio.0020383
    def __init__(self,
                 natural_state: SimGrid,
                 relative_range_thresholds=None,
                 risk_weights=np.array([-12, -3, -1, 0, 1]),
                 current_state: SimGrid = None,
                 relative_pop_thresholds=np.array([0.10,  # CR / EX: 0
                                                   0.30,  # EN: 1
                                                   0.50,  # VU: 2
                                                   0.60  # NT: 3
                                                   # LC: 4
                                                   ]),

                 relative_protected_range_thresholds=np.array([0.10,  # CR / EX: 0
                                                               0.30,  # EN: 1
                                                               0.50,  # VU: 2
                                                               0.60  # NT: 3
                                                               # LC: 4
                                                               ]),

                 epsilon=0.5,
                 # if eps=1 running_reward = last change
                 # eps = 0.5 rolling average
                 # eps < 0.5 longer legacy of long-term change
                 sufficient_protection=0.5,
                 starting_rl_status=None,
                 pop_decrease_threshold=0.01,
                 evolve_status=True,
                 min_individuals_cell=None,
                 compare_to_init_state=False,
                 ):
        super().__init__(natural_state,
                         relative_range_thresholds,
                         risk_weights,
                         evolve_status=evolve_status,
                         starting_rl_status=starting_rl_status,
                         min_individuals_cell=min_individuals_cell)
        if current_state is None:
            current_state = natural_state

        self._natural_pop_size = natural_state.individualsPerSpecies(self._min_individuals_cell)
        self._natural_range_size = natural_state.geoRangePerSpecies()
        self.current_protected_sp_range = current_state.protectedRangePerSpecies()
        self._init_range_size = natural_state.geoRangePerSpecies()

        self.current_protected_pop_sizes = current_state.protectedIndPerSpecies(self._min_individuals_cell)
        self.current_sp_range = current_state.geoRangePerSpecies()
        self.current_pop_sizes = current_state.individualsPerSpecies(self._min_individuals_cell)
        self._init_pop_size = current_state.individualsPerSpecies(self._min_individuals_cell)
        # self.species_status = np.ones(natural_state._n_species) * self.n_labels
        if self.n_labels != 5:
            sys.exit("")
        self.epsilon = epsilon
        self.sufficient_protection = sufficient_protection
        self.population_change = np.zeros(natural_state._n_species)
        self.population_change_to_init = np.zeros(natural_state._n_species)
        self._pop_change_threshold = pop_decrease_threshold
        self.relative_pop_thresholds = relative_pop_thresholds
        self._counter = 0
        self.compare_to_init_state = compare_to_init_state
        self.relative_protected_range_thresholds = relative_protected_range_thresholds
        # criterion A
        # if

        # B1, assuming 1 cell = 50x50 km (0.5 degrees)
        """
        if range_size < 1 -> CR
        if range_size < 2 -> EN


        """
        if starting_rl_status is None:
            sys.exit("\nStarting RL status must be provided for ExtinctioRiskProtectedRange object!\n ")
        else:
            self.current_rl_status = starting_rl_status + 0


    def update_pop_sizes(self, state: SimGrid):
        _ = self.classify_species(state)


    def classify_species(self, state: SimGrid):
        rl = self.current_rl_status + 0
        # update prms
        current_pop_sizes = state.individualsPerSpecies(self._min_individuals_cell)
        current_sp_range = state.geoRangePerSpecies()
        current_protected_pop_sizes = state.protectedIndPerSpecies(self._min_individuals_cell) / current_pop_sizes
        """compare to initial situation"""
        population_change_to_init = current_pop_sizes / self._init_pop_size - 1
        population_change_to_natural = current_pop_sizes / self._natural_pop_size - 1
        # print("Before", rl)
        # print("population_change_to_natural", population_change_to_natural)
        # print("population_change_to_init", population_change_to_init)

        # only update if grid moved one step (so multiple calls without steps return
        # same rl status
        if self.evolve_status:
            # TODO: using empirical RL classification as a starting point
            # TODO: ratio between protected range and init range can upgrade RL status (to a degree)
            current_protected_range_sizes = "BOH"
            range_ratio_protected = current_protected_range_sizes / self._init_range_size
            rl = np.digitize(range_ratio_protected, bins=self.relative_protected_range_thresholds)
            # print("self.current_rl_status", self.current_rl_status)

            # downgrade if pop size decreased compared to initial pop
            indx = np.where(
                (population_change_to_init < -self._pop_change_threshold) &
                (population_change_to_natural < -self._pop_change_threshold)
            )[0]
            # print("indx", indx, self._pop_change_threshold, population_change_to_init[indx], population_change_to_natural[indx])
            rl[indx] = rl[indx] - 1
            rl[rl < 0] = 0

            # upgrade is increased pop size compared to init
            indx = np.where(population_change_to_init > self._pop_change_threshold)[0]
            # print("indx + ", indx, self._pop_change_threshold, population_change_to_init[indx])
            rl[indx] = rl[indx] + 1
            rl[rl > 4] = 4
            # print("new rl                ",    rl, current_pop_sizes[7], self._natural_pop_size[7])
            self.current_rl_status = rl + 0

        # print("After:", rl)

        return rl



































