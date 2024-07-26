import sys
import numpy as np
from scipy import ndimage
np.set_printoptions(suppress=True, precision=3)  # prints floats, no scientific notation
from ..utilities import metrics

class FeaturesObservation(object):
    """data structure to collect features observation"""

    def __init__(
        self,
        quadrant_coords_list,
        sp_quadrant_list,
        protected_species_list,
        stats_quadrant,
        min_pop_requirement=None,
        feature_names=None,
        protected_quadrants=None
    ):
        self.quadrant_coords_list = quadrant_coords_list
        self.sp_quadrant_list = sp_quadrant_list
        self.protected_species_list = protected_species_list
        self.stats_quadrant = stats_quadrant
        self.min_pop_requirement = min_pop_requirement
        self.feature_names = feature_names
        self.protected_quadrants = protected_quadrants

    def getCellList(self, quadrantIndex):
        return self.quadrant_coords_list[quadrantIndex]


def get_quadrant_coord_species_clean(
    grid_size,
    sp_hist,
    resolution,
    protection_matrix=[],
    sp_threshold=1,
    error=0,
    climate_layer=[],
    climate_disturbance=0,
    pop_size_per_unit=False,
    flattened=False,
    sp_quadrant_list_arg=None,
):
    if flattened:
        grid_shape = np.array([sp_hist.shape[1], 1])
        # print(grid_shape)
        resolution_grid_size = grid_shape / resolution
        resolution_grid_size[1] = 1
        x_coord = np.arange(0, grid_shape[0] + 1, resolution[0])
        y_coord = np.array([0, 1])
        # ----
        """ if PUs == cells """
        quadrant_coords_list = []

        if sp_quadrant_list_arg is None:
            sp_quadrant_list = [
                np.where(sp_hist[:, i, 0] >= sp_threshold)
                for i in range(sp_hist.shape[1])
            ]
        else:
            sp_quadrant_list = sp_quadrant_list_arg

        """
        a = np.random.randint(0, 3, (10, 5)) # sp x cells |-> sp_hist
        b = np.random.randint(0, 2, (5)) # cells |-> protection_matrix.flatten()
        i = np.where(a > 0) # (sp x cells, 2)
        i[0][np.where(b[i[1]] == 1)[0]]
        """
        # a = sp_hist[:,:,0]
        # i = np.where(a > 0)
        # b = protection_matrix.flatten()
        # pr_sp = i[0][np.where(b[i[1]] == 1)[0]]
        # protected_species_list = np.unique(pr_sp)
        tmp = np.einsum("sij,ij->sij", sp_hist, protection_matrix)
        tmp2 = np.einsum("sij->s", tmp)
        protected_species_list = np.where(tmp2 > 0)[0]

        protected_list = protection_matrix.flatten()
        protected_list[protected_list < 1] = 0

        if climate_disturbance:
            climate_disturbance_list = climate_layer.flatten()
        else:
            climate_disturbance_list = np.zeros(sp_hist.shape[1])
        total_pop_size = np.einsum("sij->ij", sp_hist).flatten()

        l1 = [
            quadrant_coords_list,
            sp_quadrant_list,
            protected_list,
            protected_species_list,
            climate_disturbance_list,
            total_pop_size,
        ]
        return l1

        # ----
    else:
        resolution_grid_size = grid_size / resolution
        x_coord = np.arange(0, grid_size + 1, resolution[0])
        y_coord = np.arange(0, grid_size + 1, resolution[1])

    hist = sp_hist + 0
    # hist[hist >sp_threshold] = 1
    hist[hist < sp_threshold] = 0
    sp_quadrant_list = []
    quadrant_coords_list = []
    protected_list = []
    protected_species_list = []
    climate_disturbance_list = []
    total_pop_size = []
    sp_pop_quadrant_list = []

    # remove species due to error (global)
    # print(np.sum(hist), "before")
    # if error:
    #     temp = np.einsum('sij->s' ,hist)
    #     ind_observed_sp = np.random.choice(np.arange(sp_hist.shape[0]), int((1 - error) * len(temp[temp > 0])),
    #                      p=temp / np.sum(temp), replace=False)
    #     z = np.zeros(sp_hist.shape[0]).astype(int)
    #     z[ind_observed_sp] = 1
    #     hist = np.einsum('sij,s->sij', hist, z)
    #     # print(np.sort(ind_observed_sp))
    # # print(np.sum(hist),"after", hist.shape)

    for x_i in np.arange(0, int(resolution_grid_size[0])):
        for y_i in np.arange(0, int(resolution_grid_size[1])):
            Xs = np.arange(x_coord[x_i], x_coord[x_i + 1])
            Ys = np.arange(y_coord[y_i], y_coord[y_i + 1])
            quadrant_coords = np.meshgrid(Xs, Ys)
            # find which species live in range
            hist_in_quadrant = hist[:, quadrant_coords[0], quadrant_coords[1]]
            temp = np.einsum("sij->s", hist_in_quadrant)
            if error and np.sum(temp) > 0:
                # error applied per quadrant (missing rare species)
                ind_observed_sp = np.random.choice(
                    np.arange(sp_hist.shape[0]),
                    int((1 - error) * len(temp[temp > 0])),
                    p=temp / np.sum(temp),
                    replace=False,
                )
                z = np.zeros(sp_hist.shape[0]).astype(int)
                z[ind_observed_sp] = 1
                temp_1 = np.einsum("s,s->s", temp, z)
                temp_1[temp_1 > 0] = 1
                #
                # add mis-identification error (this is a fraction of the true number of species in the
                # quadrant. Mis-identification will make some of the true species disappear and some of
                # the species which are not there to be counted in. This is done independently of the
                # overall or local rarity of the species.
                # print(temp_1)
                temp_2 = np.abs(
                    temp_1
                    - np.random.binomial(
                        1, error * (np.sum(temp_1) / len(temp_1)), len(temp_1)
                    )
                )
                # print(temp_2)
                sp_in_quadrant = np.arange(sp_hist.shape[0])[temp_2 > 0]
                # print(np.sum(temp_1), np.sum(temp_2))
                # print(len(sp_in_quadrant), len(temp[temp>0]), len(sp_in_quadrant)/len(temp[temp>0]),
                #       np.sum(np.abs(temp_2-temp_1))/len(temp_2))
                # quit()

            else:
                temp = np.einsum("sij->s", hist_in_quadrant)
                temp_2 = temp
                sp_in_quadrant = np.arange(sp_hist.shape[0])[temp > 0]

            sp_quadrant_list.append(sp_in_quadrant)
            quadrant_coords_list.append([Xs, Ys])
            if len(protection_matrix) != 0:
                mean_protection = np.max(
                    protection_matrix[quadrant_coords[0], quadrant_coords[1]]
                )
                protected_list.append(mean_protection)
                if mean_protection > 0:
                    protected_species_list = protected_species_list + list(
                        sp_in_quadrant
                    )
            # else:
            #    protected_list.append( 0 )
            if climate_disturbance:
                climate_disturbance_list.append(
                    np.mean(climate_layer[quadrant_coords[0], quadrant_coords[1]])
                )
            else:
                climate_disturbance_list.append(0)
            total_pop_size.append(np.sum(temp))
            sp_pop_quadrant_list.append(temp_2)

    protected_species_list = np.unique(protected_species_list)
    if pop_size_per_unit:
        return [
            quadrant_coords_list,
            sp_pop_quadrant_list,
            protected_list,
            protected_species_list,
            climate_disturbance_list,
            total_pop_size,
        ]
    else:
        return [
            quadrant_coords_list,
            sp_quadrant_list,
            protected_list,
            protected_species_list,
            climate_disturbance_list,
            total_pop_size,
        ]


def get_thresholds(coeffs, stretch=0.2):
    # logistic function
    # return 1 / (1 + np.exp(-coeffs))
    # return np.abs(np.sin(coeffs))
    return 1 / (1 + np.exp(-stretch * coeffs))
    # return coeffs


def get_thresholds_reverse(thresh, stretch=0.2):
    # logistic function
    # return 1 / (1 + np.exp(-coeffs))
    # return np.abs(np.sin(coeffs))
    return np.log(1 / thresh - 1) / -stretch
    # return coeffs


# TODO: cleanup to rm all unused features


def get_feature_indx(mode, print_obs_mode=False):
    mode_list = [
        "protected-only",  # 0
        "full-species-monitoring",  # 1
        "citizen-science-species-monitoring",  # 2
        "one-time-full-species-monitoring",  # 3
        "value-monitoring",  # 4
        "area-monitoring",  # 5
        "return_deltaVC_sp",  # 6
        "return_deltaVC_val",  # 7
        "comb-value",  # 8
        "all",  # -1
    ]
    criterion = mode_list[mode]
    climate_features = 23
    non_protected_all = 3
    non_protected_rare = 4
    all_sp = 0
    all_rare = 1
    decrease_pop_size = 21
    budget_cost = 15
    cost = 14
    non_protected_rare_value = 17
    non_protected_value = 16
    decrease_pop_size_value = 22
    mean_delta_pop_size = 12
    mean_delta_range_size = 13
    already_protected = -1
    comb_pop_feature = 24
    tot_popsize = 25
    comb_pop_value = 26
    delta_VC_species = 27  # number of non-protected species in quadrant / cost
    delta_VC_value = 28  # value of non-protected species in quadrant / cost
    
    if criterion == "protected-only":  # 0
        indx = [already_protected]
    elif criterion == "full-species-monitoring":  # 1
        indx = [
            mean_delta_pop_size,
            non_protected_all,
            non_protected_rare,
            cost,
            already_protected,
        ]
    elif criterion == "citizen-science-species-monitoring":  # 2
        indx = [non_protected_all, cost, already_protected]
    elif criterion == "one-time-full-species-monitoring":  # 3
        indx = [non_protected_rare, non_protected_all, cost, already_protected]
    elif criterion == "value-monitoring":  # 4
        indx = [non_protected_value, non_protected_rare_value, cost, already_protected]
    elif criterion == "area-monitoring":  # 5
        indx = [cost, already_protected]
    elif criterion == "return_deltaVC_sp":  # 6
        indx = [delta_VC_species]
    elif criterion == "return_deltaVC_val":  # 7
        indx = [delta_VC_value]
    elif criterion == "comb-value":  # 8
        indx = [
            non_protected_value,
            comb_pop_value,
            cost,
            budget_cost,
            already_protected,
        ]
    elif criterion == "all":
        indx = range(27)
    else:
        sys.exit("\nError: Observe mode not found!\n")
    
    if print_obs_mode:
        print("Monitoring policy:", criterion)
        return

    return np.array(indx)


def extract_features(grid_obj,
                     grid_obj_previous,
                     quadrant_resolution,
                     current_protection_matrix,
                     rare_sp_qntl,
                     smallrange_sp_qntl=0.1,
                     cost_quadrant=[],
                     mode=[],
                     budget=0,
                     sp_threshold=1,
                     sp_values=[],
                     zero_protected=0,
                     observe_error=0,
                     flattened=False,
                     min_pop_requirement=None,
                     met_prot_target=None,
                     sp_quadrant_list_arg=None,
                     verbose=0,
                     ):
    # print("doing extract_features")
    "mode arg is used to subset features"
    grid_length = grid_obj.length
    grid_h = grid_obj.h
    # current_protection_matrix = grid_obj.protection_matrix

    # extract current features
    pop_sizes = grid_obj.individualsPerSpecies() + 1  # to avoid nan in extinct species
    range_sizes = grid_obj.geoRangePerSpecies() + 1

    # extract past features (could be saved in memory instead)
    pop_sizes_previous = grid_obj_previous.individualsPerSpecies() + 1
    range_sizes_previous = grid_obj_previous.geoRangePerSpecies() + 1

    if len(sp_values) == 0:
        sp_values = np.ones(grid_obj._n_species)
    total_value = np.sum(sp_values)

    # extract temporal variation
    # negative if declining population (min value = 0/n - 1 = -1
    delta_pop_size = pop_sizes / pop_sizes_previous - 1
    delta_range_size = range_sizes / range_sizes_previous - 1

    # extract relative change in pop size (ie relative to the total)
    delta_pop_size_rel = (pop_sizes / np.sum(pop_sizes)) / (
        pop_sizes_previous / np.sum(pop_sizes_previous)
    ) - 1

    # rare_sp_qntl =  0.1 # -> bottom 10% is considered rare
    # smallrange_sp_qntl  = 0.1 # -> 10% of total area is considered small range
    smallrange_sp_threshold = smallrange_sp_qntl * (grid_length ** 2)

    res = get_quadrant_coord_species_clean(
        grid_length,
        grid_h,
        resolution=quadrant_resolution,
        protection_matrix=current_protection_matrix,
        sp_threshold=sp_threshold,
        error=observe_error,
        climate_layer=grid_obj._climate_layer,
        climate_disturbance=grid_obj._climate_as_disturbance,
        flattened=flattened,
        sp_quadrant_list_arg=sp_quadrant_list_arg,
    )

    [
        quadrant_coords_list,
        sp_quadrant_list,
        protected_list,
        protected_species_list,
        climate_disturbance,
        total_pop_size,
    ] = res
    # k rarest species
    k = np.max([1, np.round(grid_obj.numberOfSpecies() * rare_sp_qntl).astype(int) - 1])
    pop_sizes_mod = pop_sizes + 0
    if len(protected_species_list):
        # alter pop sizes of already protected (so they are no longer considered rare)
        pop_sizes_mod[protected_species_list] = np.max(pop_sizes_mod)
        # alter pop sizes of extinct species (so they are no longer considered rare)
        try:
            pop_sizes_mod[grid_obj.extinctSpeciesIndexID()] = np.max(pop_sizes_mod)
        #pop_sizes_mod[[i for i in grid_obj._species_id_indx if grid_obj._species_id[i] in grid_obj.extinctSpeciesID()]] = np.max(pop_sizes_mod)#KD
        except:
            # for back compatibility
            pop_sizes_mod[grid_obj.extinctSpeciesID()] = np.max(pop_sizes_mod)
    idx = np.argpartition(pop_sizes_mod, k)
    idx_k_rarest_species = [idx[: k + 1]]
    # print(k, idx, idx_k_rarest_species)

    # ---
    if met_prot_target is not None:
        protected_species_list = met_prot_target
    elif min_pop_requirement is not None and len(protected_species_list) > 0:
        if verbose:
            print("Min population threshold:", min_pop_requirement)
        pop_sizes = grid_obj.protectedIndPerSpecies()
        popsize_protected_sp = pop_sizes[protected_species_list]

        # for i in range(4):
        #     diff_from_min_threshold = np.ones(1)
        #     #while np.min(diff_from_min_threshold) > 0:
        #     min_pop_requirement = min_pop_requirement * 1.01
        #     print("increased threshold:", min_pop_requirement)

        if len(protected_species_list) == len(min_pop_requirement):
            # if all species are protected and meet the threshold increase threshold
            diff_from_min_threshold = np.ones(1)
            while np.min(diff_from_min_threshold) > 0:
                min_pop_requirement = min_pop_requirement * 1.01
                popsize_protected_sp = pop_sizes[protected_species_list]
                diff_from_min_threshold = (
                    popsize_protected_sp - min_pop_requirement[protected_species_list]
                )
            if verbose:
                print("increased threshold:", min_pop_requirement)
        else:
            diff_from_min_threshold = (
                popsize_protected_sp - min_pop_requirement[protected_species_list]
            )
            if verbose:
                print(
                    diff_from_min_threshold,
                    grid_obj.protectedIndPerSpecies() / min_pop_requirement,
                )

        protected_species_list = protected_species_list[diff_from_min_threshold >= 0]
        if verbose:
            print("PROTECTED SP", len(protected_species_list))
        # ratio = popsize_protected_sp / min_pop_requirement[protected_species_list]
        # print('min_pop_requirement',tmp, len(protected_species_list), np.max(ratio))
        if verbose:
            print(len(popsize_protected_sp), len(protected_species_list))

        test = 0
        if test:
            min_pop_requirement = np.array([12, 150, 5, 10, 20])
            pop_sizes = np.array([100, 100, 2, 11, 30])
            protected_species_list = np.array([1, 2, 4])
            popsize_protected_sp = pop_sizes[protected_species_list]
            diff_from_min_threshold = (
                popsize_protected_sp - min_pop_requirement[protected_species_list]
            )
            protected_species_list = protected_species_list[diff_from_min_threshold > 0]
            #
            min_pop_requirement = np.array([12, 150, 5, 10, 20])
            pop_sizes = np.array([160, 160, 22, 21, 30])
            protected_species_list = np.array([1, 2, 4])
    # ---
    # print(rare_sp_qntl, np.max(pop_sizes))
    rare_sp_threshold = np.exp(rare_sp_qntl * np.log(np.max(pop_sizes)))
    # print(rare_sp_threshold)

    all_features_by_quadrant = list()

    counter = 0
    for i in sp_quadrant_list:
        list_features_by_quadrant = list()

        "SPECIES COUNTS"
        # 0. number of species in quadrant
        list_features_by_quadrant.append(len(i))

        # 1. number of rare species in quadrant
        list_features_by_quadrant.append(np.sum(pop_sizes[i] < rare_sp_threshold))

        # 2. number of small range species in quadrant
        list_features_by_quadrant.append(
            np.sum(range_sizes[i] < smallrange_sp_threshold)
        )

        "NON-PROTECTED SPECIES COUNTS"
        # 3. number of non-protected species in quadrant
        i_at_risk = np.setdiff1d(i, protected_species_list)  # non-protected species IDs
        list_features_by_quadrant.append(len(i_at_risk))

        # 4. number of non-protected rare species in quadrant
        # print('k', rare_sp_qntl, len(np.intersect1d(i_at_risk, idx_k_rarest_species)), len(i_at_risk))
        k_rarest_non_protected = np.intersect1d(i_at_risk, idx_k_rarest_species)
        list_features_by_quadrant.append(len(k_rarest_non_protected))
        # print(len(np.intersect1d(i_at_risk, idx_k_rarest_species)), np.intersect1d(i_at_risk, idx_k_rarest_species))
        # list_features_by_quadrant.append(np.sum(pop_sizes[i_at_risk] < rare_sp_threshold))
        # if np.sum(pop_sizes[i_at_risk] < rare_sp_threshold):
        #     print(rare_sp_threshold, np.sum(pop_sizes[i_at_risk] < rare_sp_threshold), len(i_at_risk))

        # 5. number of non-protected small range species in quadrant
        list_features_by_quadrant.append(
            np.sum(range_sizes[i_at_risk] < smallrange_sp_threshold)
        )

        "TEMPORAL FEATURES (ALL SPECIES)"  # TODO: remove?
        # 6. number of species with decreased pop size
        list_features_by_quadrant.append(np.sum(delta_pop_size[i] < 0))

        # 7. number of species with decreased range size
        list_features_by_quadrant.append(np.sum(delta_range_size[i] < 0))

        # 8. delta pop size
        list_features_by_quadrant.append(np.log(0.01 + np.sum(1 + delta_pop_size[i])))

        # 9. delta range size
        list_features_by_quadrant.append(np.log(0.01 + np.sum(1 + delta_range_size[i])))

        "TEMPORAL FEATURES (NON-PROTECTED SPECIES)"
        # 10. number of non-protected species with decreased pop size
        list_features_by_quadrant.append(np.sum(delta_pop_size[i_at_risk] < 0))

        # 11. number of non-protected species with decreased range size
        list_features_by_quadrant.append(np.sum(delta_range_size[i_at_risk] < 0))

        if len(i_at_risk):
            # 12. delta pop size in non-protected species
            list_features_by_quadrant.append(np.mean(delta_pop_size[i_at_risk]))
            # list_features_by_quadrant.append(np.log(0.01 + np.sum(1 + delta_pop_size[i_at_risk])))

            # 13. delta range size in non-protected species
            list_features_by_quadrant.append(np.mean(delta_range_size[i_at_risk]))
            # list_features_by_quadrant.append(np.log(0.01 + np.sum(1 + delta_range_size[i_at_risk])))
        else:
            # 0 means no change in pop size
            list_features_by_quadrant.append(0)
            list_features_by_quadrant.append(0)
        "COSTS"
        # 14. additional protection cost
        if len(cost_quadrant) > 0:
            cost_q = cost_quadrant[counter]
        else:
            cost_q = 0
        # list_features_by_quadrant.append(cost_q)
        # set to 0 is cost > budget or if area already protected
        # when area is protected the disturbance is 0 and the cost is otherwise set to the baseline (e.g. 5)
        if cost_q > budget or protected_list[counter] == 1:
            list_features_by_quadrant.append(0)
        else:
            list_features_by_quadrant.append(cost_q)

        # 15. budget left minus cost
        list_features_by_quadrant.append(budget - cost_q)

        "SPECIES VALUES"  # Changes depending on what is being used as a reward
        # 16. value of non-protected species in quadrant
        non_protected_value = np.sum(sp_values[i_at_risk])
        list_features_by_quadrant.append(non_protected_value)

        # 17. value of non-protected rare species in quadrant
        # i_rare = i_at_risk[pop_sizes[i_at_risk] < rare_sp_threshold]
        list_features_by_quadrant.append(np.sum(sp_values[k_rarest_non_protected]))

        # 18. value of sp with decreased pop size
        i_decreasing = i_at_risk[delta_pop_size[i_at_risk] < 0]
        list_features_by_quadrant.append(np.sum(sp_values[i_decreasing]))

        # 19. value of non-protected small range species in quadrant
        i_small = i_at_risk[range_sizes[i_at_risk] < smallrange_sp_threshold]
        list_features_by_quadrant.append(np.sum(sp_values[i_small]))

        # 20. value of sp with decreased range size
        i_smaller = i_at_risk[delta_range_size[i_at_risk] < 0]
        list_features_by_quadrant.append(np.sum(sp_values[i_smaller]))

        "RELATIVE POP SIZE CHANGE"
        # 21. delta_pop_size_rel
        # indx = np.arange(grid_obj._n_species)[delta_pop_size_rel < 0]
        # i_rel_decreasing = np.intersect1d(i_at_risk, indx )
        list_features_by_quadrant.append(
            len(i_at_risk[delta_pop_size_rel[i_at_risk] < 0])
        )
        # print(counter, len(i_at_risk[delta_pop_size_rel[i_at_risk] < 0]), len(i_at_risk)) #np.log10(pop_sizes[i_at_risk]))
        # print(delta_pop_size_rel[i_at_risk])

        "value +rel change"
        # 22. rare delta_pop_size_rel
        list_features_by_quadrant.append(
            np.sum(sp_values[i_at_risk[delta_pop_size_rel[i_at_risk] < 0]])
        )
        # print(np.sum(sp_values[i_at_risk[delta_pop_size_rel[i_at_risk] < 0]]))

        "CLIMATE"
        # 23. climate disturbance
        list_features_by_quadrant.append(climate_disturbance[counter])

        "COMBINED"
        # 24. combined
        f1 = i_at_risk[delta_pop_size_rel[i_at_risk] < 0]
        f2 = i_at_risk[pop_sizes[i_at_risk] < np.max(pop_sizes) * rare_sp_qntl]
        # print(np.intersect1d(f1,f2))
        # delta_pop_size_rel[i_at_risk] ** np.log10(pop_sizes[i_at_risk])
        # if budget - cost_q > 0:
        list_features_by_quadrant.append(len(np.intersect1d(f1, f2)))
        # else:
        #     list_features_by_quadrant.append(0)

        # 25. overall pop change (compared to step 0)
        list_features_by_quadrant.append(total_pop_size[counter])

        # 26. combined - value
        list_features_by_quadrant.append(np.sum(sp_values[np.intersect1d(f1, f2)]))

        "deltaVC values"
        # 27. number of non-protected species in quadrant / cost
        delta_den = cost_q / np.mean(cost_quadrant)
        rel_sp = len(i_at_risk) / grid_obj._n_species
        # print(cost_q)
        list_features_by_quadrant.append(rel_sp / delta_den)

        # list_features_by_quadrant.append((len(i_at_risk) / grid_obj._n_species) / delta_den)

        # 28 value of non-protected species in quadrant / cost
        list_features_by_quadrant.append(
            (non_protected_value / total_value) / delta_den
        )

        # LAST. protection
        # print('protected_list',protected_list)
        list_features_by_quadrant.append(
            protected_list[counter]
        )  # 1: protected, 0: non protected

        # print("list_features_by_quadrant", list_features_by_quadrant)

        list_features_by_quadrant = np.array(list_features_by_quadrant)
        if zero_protected:
            list_features_by_quadrant *= 1 - protected_list[counter]
        all_features_by_quadrant.append(list_features_by_quadrant)

        counter += 1

    all_features_by_quadrant = np.array(all_features_by_quadrant)
    # all_features_by_quadrant_original = all_features_by_quadrant + 0
    # normalizer = np.array([# species features
    #                        grid_obj._n_species, grid_obj._n_species, grid_obj._n_species,
    #                        grid_obj._n_species / 2, grid_obj._n_species / 2, grid_obj._n_species / 2,
    #                        # tempora features (ratios)
    #                        1, 1, 1, 1,
    #                        1, 1, 1, 1,
    #                        len(sp_quadrant_list)*0.1, len(sp_quadrant_list)*0.1, # <- 14, 15
    #                        # value features
    #                        total_value, total_value / 2, total_value / 2,
    #                        total_value / 2, total_value / 2,
    #                        # relative pop size change
    #                        grid_obj._n_species / 2, grid_obj._n_species / 2,
    #                        # climate
    #                        1,
    #                        grid_obj._n_species, 1000*(quadrant_resolution[0]*quadrant_resolution[1]),grid_obj._n_species,
    #                        1, 1, # deltaVC
    #                        1
    #                        ])
    # all_features_by_quadrant = np.log(np.exp(all_features_by_quadrant) + 1)

    normalizer = np.max(all_features_by_quadrant, axis=0) - np.min(
        all_features_by_quadrant, axis=0
    )  # np.std(all_features_by_quadrant, axis=0)
    normalizer[-1] = 1
    normalizer[normalizer == 0] = 1
    # print('normalizer', normalizer[get_feature_indx(mode)])
    # print(np.max(all_features_by_quadrant, axis=0)[get_feature_indx(mode)], np.min(all_features_by_quadrant, axis=0)[get_feature_indx(mode)])

    # all_features_by_quadrant /= normalizer  # + 0.1
    # all_features_by_quadrant = (all_features_by_quadrant - np.mean(all_features_by_quadrant, axis=0)) / normalizer

    # MIN-MAX rescaler
    r = (
        all_features_by_quadrant - np.min(all_features_by_quadrant, axis=0)
    ) / normalizer
    all_features_by_quadrant = r
    # print('MIN', np.min(all_features_by_quadrant, axis=0))
    # print('MAX', np.max(all_features_by_quadrant, axis=0))
    # print("cost", all_features_by_quadrant[:, 14])

    # all_features_by_quadrant_original = all_features_by_quadrant_original[:, get_feature_indx(mode)]
    all_features_by_quadrant = all_features_by_quadrant[:, get_feature_indx(mode)]
    # print(all_features_by_quadrant[:5,:])
    features = FeaturesObservation(
        quadrant_coords_list,
        sp_quadrant_list,
        protected_species_list,
        all_features_by_quadrant,
        min_pop_requirement=min_pop_requirement,
    )
    return features


############

def get_rl_features_cell(h, pm, rl, sp_threshold):

    h_non_protected = (h > sp_threshold) * (1 - pm)
    h_protected = (h > sp_threshold) * pm
    protected_species_id = np.einsum('sxy -> s', h_protected)  # shape: species

    min_number_protected_cells_sp = 1
    non_protected_species_id = protected_species_id < min_number_protected_cells_sp
    h_non_protected_elsewhere = np.einsum('sxy, s -> sxy', h_non_protected, non_protected_species_id)
    # FEATURES
    # number of species per cell per RL class
    h_rl = [np.einsum('sxy, s -> xy',
                   (h > sp_threshold).astype(int), (rl == i).astype(int)).flatten() for i in range(4)]

    h_rl_not_protected = [np.einsum('sxy, s -> xy',
                   (h_non_protected_elsewhere >= 1).astype(int),
                   (rl == i).astype(int)).flatten() for i in range(4)]

    return np.array(h_rl + h_rl_not_protected).T, protected_species_id



##############

def get_feature_restore_indx(mode="carbon", print_obs_mode=False):
    if mode == "carbon":
        indx = np.array([1, 7])
    elif mode == "ext_risk":
        indx = np.array([0, 2, 3, 4, 7])
    elif mode == "ext_risk_carbon":
        indx = np.array([0, 1, 2, 3, 4, 7])
    elif mode == "all":
        indx = np.arange(9)
    elif mode == "ext_risk_protect":
        indx = np.array([0, 2, 3, 4, 5, 7])
    elif mode == "star_t":
        indx = np.array([8])
    elif mode == "sp_risk_protect":
        indx = np.arange(8)
    else:
        sys.exit("Mode not found: %s" % mode)

    if print_obs_mode:
        print(mode, indx.shape, indx)

    return indx


def extract_features_restore(grid_obj,
                             grid_obj_previous,
                             quadrant_resolution, # env.resolution
                             current_protection_matrix,
                             species_threat_label, # env.getExtinction_risk_labels()
                             n_threat_labels=5, # env.species_risk_criteria.n_labels
                             cost_quadrant=None, # env.getProtectCostQuadrant()
                             quandrant_grid_indx=None, # env._quandrant_grid_indx
                             sp_threshold=1,
                             mode=0, # currently not used could change features depending on reward
                             feature_set = "all",
                             use_true_natural_state=True, # if False approximates natural diversity and carbon
                             observe_error=0,
                             budget=0,
                             min_pop_requirement=None,
                             flattened=False,
                             verbose=0,
                             normalize=True,
                             get_protected_species_list=False
                             ):
    if quandrant_grid_indx is None:
        quandrant_grid_indx = get_quadrant_indx_grid(grid_obj.length, quadrant_resolution)

    if feature_set == "sp_risk_protect":

        feat_list, protected_species_id = get_rl_features_cell(grid_obj_previous.h, # based on potential diversity
                                                               current_protection_matrix,
                                                               species_threat_label,
                                                               sp_threshold)

        feat_names = np.array([
                              "CR",
                              "EN",
                              "VU",
                              "NT",
                              "CR_not_protected",
                              "EN_not_protected",
                              "VU_not_protected",
                              "NT_not_protected",
                              ])

        protection_vec = current_protection_matrix.flatten()


    else:

        sp_ind_quadrant = get_sp_indx_per_quadrant(grid_obj.h, quandrant_grid_indx)


        n_features = (1 + 1 + n_threat_labels + 1 + 1 + 1)
        feat_list = np.zeros((len(quandrant_grid_indx[1]), n_features))
        feat_names= np.array(["Potential species richness",  # 0
                              "Potential carbon",            # 1
                              "CR species (relative count)", # 2
                              "EN species (relative count)", # 3
                              "VU species (relative count)", # 4
                              "NT species (relative count)", # 5
                              "LC species (relative count)", # 6
                              "Budget - cost",               # 7
                              "STAR-t",                      # 8
                              # "Protected or restored"
                              ])

        s_current = get_sp_count_per_quadrant(grid_obj.h, quandrant_grid_indx, threshold=sp_threshold)
        if use_true_natural_state:
            s_natural = get_sp_count_per_quadrant(grid_obj_previous.h, quandrant_grid_indx, threshold=sp_threshold)
        else:
            dist_q = get_mean_grid_value_quadrant(grid_obj.disturbance_matrix,
                                                   quandrant_grid_indx)
            s_natural = (s_current / (1 - dist_q))

        feat_list[:,0] = (
            # 0: potential species richness
            (s_natural - s_current) / np.mean(0.0001 + np.abs(s_natural - s_current))
        )

        c_natural = get_carbon_quadrant(grid_obj_previous, quandrant_grid_indx)
        c_current = get_carbon_quadrant(grid_obj, quandrant_grid_indx)

        #- remove 0-carbon cells (e.g. marine)
        i_tmp = np.where(c_natural > c_current)[0]

        if len(i_tmp):
            feat_list[i_tmp,1] = (
                    # 1: potential carbon
                    (c_natural[i_tmp] - c_current[i_tmp]) / np.mean(0.0001 + np.abs(c_natural[i_tmp] - c_current[i_tmp]))
            )

        # print("species_threat_label", species_threat_label)
        # tmp = get_threat_count_quadrant(
        #         species_threat_label, # env.getExtinction_risk_labels()
        #         sp_ind_quadrant=sp_ind_quadrant,
        #         n_threat_labels=5, # env.species_risk_criteria.n_labels
        #         normalize=True)
        # print(tmp.shape, feat_list.shape)
        feat_list[:,np.arange(2,2+n_threat_labels)] = (
            # 2-6: n. species per Threat class
            get_threat_count_quadrant(
                species_threat_label, # env.getExtinction_risk_labels()
                sp_ind_quadrant=sp_ind_quadrant,
                n_threat_labels=n_threat_labels, # env.species_risk_criteria.n_labels
                normalize=True)
        )

        feat_list[:,2+n_threat_labels] = (
            # 7: cost
            budget - cost_quadrant
        )

        if "STAR-t" in feat_names[get_feature_restore_indx(feature_set)]:
            # by-pass STAR calculation to speed up if not used
            star_t, _ = metrics.calc_STAR_from_grid(natural_h=grid_obj_previous.h, current_h=grid_obj.h,
                                                    quandrant_grid_indx=quandrant_grid_indx,
                                                    sp_natural_range=grid_obj_previous.geoRangePerSpecies(),
                                                    sp_ext_risk=species_threat_label,
                                                    species_presence_threshold=0.01)
            feat_list[:, 3 + n_threat_labels] = star_t.flatten()
            # print("\nstar_t\n", np.log(0.000001 + np.unique(star_t.flatten())))
            # print("STATE MONITOR: grid_obj_previous, grid_obj", np.sum(grid_obj_previous.h), np.sum(grid_obj.h))

        protection_vec = (
            # protection
            # 0 -> unprotected quadrants; 1 -> protected quadrants
            get_protected_quadrants(protection_matrix=current_protection_matrix,
                                    quandrant_grid_indx=quandrant_grid_indx)
        )

    if normalize:
        # set all features of protected areas to minimum (i.e. 0 after rescaling):
        feat_list[protection_vec == 1, :] = np.min(feat_list, 0)

        # normalize range 0-1
        feat_list = feat_list - np.mean(feat_list,0)
        den = np.max(feat_list,0) - np.min(feat_list,0)

        den[den < 0.0001] = 1
        feat_list = feat_list / den
        feat_list += np.abs(np.min(feat_list,0))

    if feature_set != "sp_risk_protect":
        all_features_by_quadrant = feat_list[:, get_feature_restore_indx(feature_set)]
        all_feat_names = feat_names[get_feature_restore_indx(feature_set)]

        protected_species_list = []
        if get_protected_species_list:
            for i in range(len(sp_ind_quadrant)):
                if protection_vec[i] == 1:
                    protected_species_list = protected_species_list + list(sp_ind_quadrant[i])
            protected_species_list = np.unique(np.array(protected_species_list))
    else:
        # print("all_features_by_quadrant", all_features_by_quadrant.shape)
        all_features_by_quadrant = feat_list
        all_feat_names = feat_names
        protected_species_list = protected_species_id
        sp_ind_quadrant = None


    features = FeaturesObservation(
        quadrant_coords_list=[],
        sp_quadrant_list=sp_ind_quadrant,
        protected_species_list=protected_species_list,
        stats_quadrant=all_features_by_quadrant,
        min_pop_requirement=min_pop_requirement,
        feature_names=all_feat_names,
        protected_quadrants=protection_vec
    )
    return features


def get_quadrant_indx_grid(grid_size, resolution=np.array([1, 1])):
    resolution_grid_size = grid_size / resolution
    x_coord = np.arange(0, grid_size + 1, resolution[0])
    y_coord = np.arange(0, grid_size + 1, resolution[1])

    grid_indx = np.zeros((grid_size, grid_size)).astype(int)

    counter = 0
    for x_i in np.arange(0, int(resolution_grid_size[0])):
        for y_i in np.arange(0, int(resolution_grid_size[1])):
            Xs = np.arange(x_coord[x_i], x_coord[x_i + 1])
            Ys = np.arange(y_coord[y_i], y_coord[y_i + 1])
            quadrant_coords = np.meshgrid(Xs, Ys)
            grid_indx[quadrant_coords[0], quadrant_coords[1]] = counter
            counter += 1

    return grid_indx

##### restore features

def get_individual_species_quadrant(grid, quandrant_grid_indx):
    # shape: (s, q)
    # quandrant_grid_indx = [l, i, l_sp, i_sp]
    return ndimage.sum(grid,
                       labels=quandrant_grid_indx[2],
                       index=quandrant_grid_indx[3])


def get_sp_count_per_quadrant(grid, quandrant_grid_indx, threshold=1):
    max_cell_individual_sp_quadrant = ndimage.maximum(grid,
                                                      labels=quandrant_grid_indx[2],
                                                      index=quandrant_grid_indx[3])
    # counting if > threshold individual in at least one cell
    return np.sum(max_cell_individual_sp_quadrant >= threshold, 0)


def get_sp_indx_per_quadrant(grid, quandrant_grid_indx, threshold=1):
    max_cell_individual_sp_quadrant = ndimage.maximum(grid,
                                                      labels=quandrant_grid_indx[2],
                                                      index=quandrant_grid_indx[3])

    l = []
    for i in range(quandrant_grid_indx[3].shape[1]):
        l.append(list(np.where(max_cell_individual_sp_quadrant[:, i] >= 0.1)[0]))

    # sp_id, quadrant_id = np.where(max_cell_individual_sp_quadrant >= threshold)
    # # l = [list(sp_id[quadrant_id == i]) for i in np.unique(quadrant_id)]
    # l = [list(sp_id[quadrant_id == i]) for i in range(quandrant_grid_indx[3].shape[1])]
    # print("[sp_id[quadrant_id == i] for i in np.unique(quadrant_id)]", len(l))
    return l

def get_carbon_quadrant(grid_obj, quandrant_grid_indx):
    return ndimage.sum(grid_obj.getCarbonValue_cell(),
                       labels=quandrant_grid_indx[0],
                       index=quandrant_grid_indx[1])


def get_protected_quadrants(protection_matrix, quandrant_grid_indx):
    return ndimage.maximum(protection_matrix,
                   labels=quandrant_grid_indx[0],
                   index=np.unique(quandrant_grid_indx[0])) > 0

def get_threat_count_quadrant(species_threat_label, # env.getExtinction_risk_labels()
                              sp_ind_quadrant, # cn.get_sp_indx_per_quadrant(env.bioDivGrid.h, env._quandrant_grid_indx)
                              n_threat_labels=5, # env.species_risk_criteria.n_labels
                              normalize=True):
    sp_threat_quadrant_count = np.array([
        np.bincount(species_threat_label[i],
                     minlength=n_threat_labels,
                     ) for i in sp_ind_quadrant
        ])
    if normalize:
        sp_threat_quadrant_count = sp_threat_quadrant_count / np.mean(sp_threat_quadrant_count)
    return sp_threat_quadrant_count

def get_mean_grid_value_quadrant(val, # any 2D array of shape = env.bioDivGrid.h[0].shape
                                 quandrant_grid_indx): # env._quandrant_grid_indx
    return ndimage.mean(val,
                        labels=quandrant_grid_indx[0],
                        index=np.unique(quandrant_grid_indx[0]))

def get_sum_grid_value_quadrant(val, # any 2D array of shape = env.bioDivGrid.h[0].shape
                                 quandrant_grid_indx): # env._quandrant_grid_indx
    return ndimage.sum(val,
                        labels=quandrant_grid_indx[0],
                        index=np.unique(quandrant_grid_indx[0]))



##### Alter knowledge of environment

def alter_init_grid(grid_obj,
                    per_species_obs_err=10, # magnitude
                    dd_fraction=0.3,
                    bias_steepness=1  # (smaller than 1 flattens probs and increases missing data)
                    ):
    # --- ADD ERROR in knowledge of the natural state
    h_tmp = grid_obj._h + 0
    # add over and under estimation of species
    multi = np.exp(np.random.uniform(
        np.log(1 / per_species_obs_err), np.log(per_species_obs_err), h_tmp.shape[0]
    ))
    h_tmp = np.einsum('sxy, s -> sxy', h_tmp, multi)
    # probability based on abundance -> rare more difficult to predict
    p_unobserved = (1 / (1 + h_tmp)) ** bias_steepness  # (species, cell, cell)
    rr = np.random.random(h_tmp.shape)
    h_tmp[rr < p_unobserved] = 0

