import numpy as np
from enum import Enum
import scipy.stats
from ..utilities.misc import get_rnd_gen

small_number = 1e-10
from numba import jit
import random


class DisturbanceGeneratorType(Enum):
    """enum to define types of disturbance generators"""

    INITIAL_RANDOM_UNIFORM = 1
    INITIAL_CONST_UNIFORM = 2
    INITIAL_GAUSSIAN = 3


class DisturbanceGeneratorFactory(object):
    @staticmethod
    def buildDisturbanceGenerator(disturbanceGeneratorType, *args, **kwargs):
        if disturbanceGeneratorType == DisturbanceGeneratorType.INITIAL_RANDOM_UNIFORM:
            return InitialRandomUniformDisturbanceGenerator(*args, **kwargs)
        elif disturbanceGeneratorType == DisturbanceGeneratorType.INITIAL_CONST_UNIFORM:
            return InitialConstUniformDisturbanceGenerator(*args, **kwargs)
        elif disturbanceGeneratorType == DisturbanceGeneratorType.INITIAL_GAUSSIAN:
            return InitialGaussianDisturbanceGenerator(*args, **kwargs)
        else:
            raise (TypeError(disturbanceGeneratorType, "not found"))


class DisturbanceGenerator(object):
    def updateDisturbance(self, disturbance_matrix):
        pass


class InitialRandomUniformDisturbanceGenerator(object):
    # random disturbance in each cell ~ U(0,1)
    def __init__(self, counter, seed=1234):
        self._counter = counter
        self._rs = get_rnd_gen(seed)

    def updateDisturbance(self, disturbance_matrix):
        if self._counter < 1:
            disturbance_matrix = self._rs.random(disturbance_matrix.shape)
            self._counter += 1
        else:
            pass
        return disturbance_matrix


class InitialConstUniformDisturbanceGenerator(object):
    # equal disturbance across cells = magnitude or ~ U(0,1)
    def __init__(self, counter, magnitude=0.5):
        self._counter = counter
        self._magnitude = magnitude

    def updateDisturbance(self, disturbance_matrix):
        if self._counter < 1:
            disturbance_matrix = np.zeros(disturbance_matrix.shape) + self._magnitude
            self._counter += 1
        else:
            pass
        return disturbance_matrix


class LinearIncreaseUniformDisturbanceGenerator(object):
    # equal disturbance across cells = magnitude or ~ U(0,1)
    def __init__(self, counter, magnitude=0.05):
        self._counter = counter
        self._magnitude = magnitude

    def updateDisturbance(self, disturbance_matrix):
        disturbance_matrix = disturbance_matrix + self._magnitude
        self._counter += 1
        disturbance_matrix[disturbance_matrix >= 1] = 1 - small_number
        return disturbance_matrix


class InitialGaussianDisturbanceGenerator(object):
    # equal disturbance across cells = magnitude or ~ U(0,1)
    def __init__(self, counter, sig=10, n_peaks=1, mean_disturbance=0, seed=1234):
        self._counter = counter
        self._sig = sig
        self._n_peaks = n_peaks
        self._mean_disturbance = mean_disturbance  # set target overall mean
        self._rs = get_rnd_gen(seed)

    def updateDisturbance(self, disturbance_matrix):
        if self._counter < 1:
            disturbance_matrix = np.zeros(disturbance_matrix.shape)
            length = disturbance_matrix.shape[0]

            indx = np.meshgrid(np.arange(length), np.arange(length))
            locsxy = self._rs.uniform(0, length, (2, self._n_peaks))
            for i in range(self._n_peaks):
                # print(locsxy[:,i])
                disturbance_matrix += scipy.stats.norm.pdf(
                    indx[0], loc=locsxy[0, i], scale=self._sig
                ) * scipy.stats.norm.pdf(indx[1], loc=locsxy[1, i], scale=self._sig)

            disturbance_matrix = disturbance_matrix / np.max(disturbance_matrix)
            if self._mean_disturbance > 0:
                exponents = np.linspace(0.05, 5, 10000)
                means = np.array([np.mean(disturbance_matrix ** i) for i in exponents])
                diff = np.abs(means - self._mean_disturbance)
                disturbance_matrix = disturbance_matrix ** exponents[np.argmin(diff)]
            # print( np.mean(disturbance_matrix) )
            self._counter += 1
            disturbance_matrix = disturbance_matrix - small_number  # avoid exactly 1
        else:
            pass
        return disturbance_matrix


class TimeLinearGaussianDisturbanceGenerator(object):
    """
    A number of Norm are initialized at the first call and their intensity is linearly
    increased through time.
    """

    # equal disturbance across cells = magnitude or ~ U(0,1)
    def __init__(self, counter, sig=10, n_peaks=1, mean_disturbance=0, n_steps=10, seed=0):
        self._counter = counter
        self._sig = sig
        self._n_peaks = n_peaks
        self._mean_disturbance = mean_disturbance  # set target overallmean
        self.n_steps = n_steps
        if seed:
            rr = seed
        else:
            rr = random.randint(1000, 9999)
        self._rr = rr
        self._rs = get_rnd_gen(rr)

    def updateDisturbance(self, disturbance_matrix):
        self._rs = get_rnd_gen(self._rr + self._counter)

        if self._counter < 1:
            disturbance_matrix = np.zeros(disturbance_matrix.shape)
            length = disturbance_matrix.shape[0]
            indx = np.meshgrid(np.arange(length), np.arange(length))
            locsxy = self._rs.uniform(0, length, (2, self._n_peaks))
            for i in range(self._n_peaks):
                # print(locsxy[:,i])
                disturbance_matrix += scipy.stats.norm.pdf(
                    indx[0], loc=locsxy[0, i], scale=self._sig
                ) * scipy.stats.norm.pdf(indx[1], loc=locsxy[1, i], scale=self._sig)

            disturbance_matrix = disturbance_matrix / np.max(disturbance_matrix)
            if self._mean_disturbance > 0:
                exponents = np.linspace(0.05, 5, 10000)
                means = np.array([np.mean(disturbance_matrix ** i) for i in exponents])
                diff = np.abs(means - self._mean_disturbance / self.n_steps)
                disturbance_matrix = disturbance_matrix ** exponents[np.argmin(diff)]
            # print( np.mean(disturbance_matrix) )
            self._counter += 1
            disturbance_matrix = disturbance_matrix - small_number  # avoid exactly 1
        else:
            if abs(np.mean(disturbance_matrix) - self._mean_disturbance) > 0.001:
                disturbance_matrix += self._mean_disturbance / self.n_steps
                disturbance_matrix[disturbance_matrix >= 1] = 1 - small_number
                if np.mean(disturbance_matrix) > self._mean_disturbance:
                    while (
                        abs(np.mean(disturbance_matrix) - self._mean_disturbance)
                        > 0.001
                    ):
                        disturbance_matrix -= 0.001

            else:
                pass

        return disturbance_matrix


class TimeIncrementalGaussianDisturbanceGenerator(object):
    """
    Norm densities are increasily added through time.
    """

    # equal disturbance across cells = magnitude or ~ U(0,1)
    def __init__(
        self,
        counter,
        sig=10,
        n_peaks=1,
        peak_disturbance=0.99,
        n_peaks_per_step=3,
        stop_at=1000,
        max_disturbance=1 - small_number,
        seed=None,
    ):
        self._counter = counter
        self._sig = sig
        self._n_peaks = n_peaks
        self._peak_disturbance = peak_disturbance  # set max disturbance
        self._n_peaks_per_step = n_peaks_per_step
        self._step = 0
        self._stop_at = stop_at
        self._max_disturbance = max_disturbance
        if seed:
            rr = seed
        else:
            rr = random.randint(1000, 9999)
        self._rr = rr
        self._rs = get_rnd_gen(rr)


    def updateDisturbance(self, disturbance_matrix):
        self._rs = get_rnd_gen(self._rr + self._counter)


        self._step += 1
        if self._step < self._stop_at:
            for i in range(self._n_peaks_per_step):
                if self._counter <= self._n_peaks:
                    # disturbance_matrix = np.zeros(disturbance_matrix.shape)
                    length = disturbance_matrix.shape[0]

                    indx = np.meshgrid(np.arange(length), np.arange(length))
                    locsxy = self._rs.uniform(0, length, (2, 1))
                    min_sig2 = 1
                    sig_tmp = self._rs.uniform(min_sig2, self._sig, 2)
                    # print("\n\nlocsxy:", locsxy, "\n")
                    disturbance_matrix_tmp = scipy.stats.norm.pdf(
                        indx[0], loc=locsxy[0, 0], scale=sig_tmp[0]
                    ) * scipy.stats.norm.pdf(
                        indx[1], loc=locsxy[1, 0], scale=sig_tmp[1]
                    )

                    disturbance_matrix_tmp = (
                        disturbance_matrix_tmp / np.max(disturbance_matrix_tmp)
                    ) * (self._rs.random() * self._peak_disturbance)

                    # scaled to the max possible disturbance (from the two narrowest normal PDFs)
                    # disturbance_matrix_tmp = disturbance_matrix_tmp / (scipy.stats.norm.pdf(0, loc=0, scale=min_sig2))**2) * self._peak_disturbance
                    disturbance_matrix = disturbance_matrix_tmp + disturbance_matrix
                    self._counter += 1
                    disturbance_matrix[
                        disturbance_matrix >= self._max_disturbance
                    ] = self._max_disturbance
                else:
                    pass

        return disturbance_matrix


@jit(nopython=True)
def get_all_to_all_dist(coord, n_cells):
    # print("calc distances...")
    linear_indx_cells = np.arange(n_cells ** 2)
    d = np.zeros((len(linear_indx_cells), len(coord), len(coord)))
    for i in linear_indx_cells:
        xa = coord[int(np.floor(i / n_cells))]
        ya = coord[np.mod(i, n_cells)]  # numpy modulo operation
        d_temp = np.zeros((len(coord), len(coord)))
        h = 0
        for x in coord:
            j = 0
            for y in coord:
                d_temp[h, j] = np.sqrt((xa - x) ** 2 + (ya - y) ** 2)
                j += 1
            h += 1
        d[i] = d_temp
    # print("done.")
    return d


def get_coordinates(coord, n_cells):
    linear_indx_cells = np.arange(n_cells ** 2)
    d = np.zeros((len(linear_indx_cells), 3))
    for i in linear_indx_cells:
        xa = int(np.floor(i / n_cells))
        ya = np.mod(i, n_cells)
        d[i, :] = np.array([i, xa, ya])
    return d


class DiffusionDisturbanceGenerator(object):
    def __init__(
        self,
        counter=0,
        extent=10,
        n_init_events=1,
        dist_resolution=10,
        expansion_rate=2,
        p_grow_disturbance=0.25,
        max_disturbance=0.95,
        seed=0,
        mask_disturbance=None # if set to a matrix will multiply disturbance after update
    ):
        self._counter = counter
        self._extent = extent  # extent of disturbance increase
        self._n_init_events = n_init_events
        self._disturbance_resolution = dist_resolution  # will be rescaled to 0-1
        self._curr_ind = list()
        self._disp_rate = expansion_rate
        self._p_grow_disturbance = (
            p_grow_disturbance  # as opposed to initializing a new start point
        )
        self._max_disturbance = max_disturbance
        if seed:
            rr = seed
        else:
            rr = random.randint(1000, 9999)
        self._rr = rr
        self._mask_disturbance = mask_disturbance
        self._rs = get_rnd_gen(rr)


    def updateDisturbance(self, disturbance_matrix):
        if self._counter == 0:
            self._length = disturbance_matrix.shape[0]
            self._coord = np.linspace(0.5, self._length - 0.5, self._length)
            self._cell_id_n_coord = get_coordinates(self._coord, self._length).astype(
                int
            )
            self._dist_matrix = get_all_to_all_dist(self._coord, self._length)
            self._curr_ind = []
            grow_disturbance = 0
        else:
            grow_disturbance = self._rs.binomial(1, self._p_grow_disturbance)

        if grow_disturbance:
            discrete_dist_per_cell = (
                self._disturbance_resolution * disturbance_matrix.flatten()
            )
            aval_space = np.ones(discrete_dist_per_cell.shape)
            tmp = 0
            for i in range(self._extent):
                for j in self._rs.choice(
                    np.arange(len(self._curr_ind)), len(self._curr_ind)
                ):
                    cell_id = self._curr_ind[j]
                    # when reached carrying capacity set respective aval_space to 0
                    aval_space[
                        discrete_dist_per_cell == self._disturbance_resolution
                    ] = 0
                    # disp = 1. / self._disp_rate  # higher dispersal rate, smaller rate of the exp distribution
                    # disp_probability = disp * np.exp(-disp * self._dist_matrix[cell_id])  # dispersal is exponentially distributed
                    disp_probability = scipy.stats.norm.pdf(
                        self._dist_matrix[cell_id], 0, self._disp_rate
                    )
                    # disp_probability = scipy.stats.cauchy.pdf(self._dist_matrix[cell_id], 0, scale=self._disp_rate)

                    disp_vec = disp_probability.flatten()
                    sampling_prob = disp_vec * aval_space

                    selected_cell = self._rs.choice(
                        self._cell_id_n_coord[:, 0],
                        p=sampling_prob / np.sum(sampling_prob),
                    )

                    # keep track of where individuals are being added
                    discrete_dist_per_cell[selected_cell] += 1
                    tmp += 1

                    # append new individual to current population
                    self._curr_ind.append(selected_cell)

                    if tmp >= self._extent:
                        break

                if tmp >= self._extent:
                    break

            disturbance_matrix = (
                discrete_dist_per_cell.reshape(disturbance_matrix.shape)
                / self._disturbance_resolution
            )

        else:
            expansion_prob = disturbance_matrix.flatten()
            if np.sum(expansion_prob) == 0:
                expansion_prob += 0.1
            curr_ind_tmp = list(
                self._rs.choice(range(self._length ** 2), self._n_init_events,
                                p=expansion_prob/np.sum(expansion_prob))
            )
            discrete_dist_per_cell_tmp = np.zeros(self._length * self._length)
            aval_space = np.ones(discrete_dist_per_cell_tmp.shape)
            for i in range(self._extent):
                for j in self._rs.choice(
                    np.arange(len(curr_ind_tmp)), len(curr_ind_tmp)
                ):
                    cell_id = curr_ind_tmp[j]
                    # when reached carrying capacity set respective aval_space to 0
                    aval_space[
                        discrete_dist_per_cell_tmp == self._disturbance_resolution
                    ] = 0
                    # disp = 1. /self._disp_rate  # higher dispersal rate, smaller rate of the exp distribution
                    # disp_probability = disp * np.exp(-disp * dist_matrix[cell_id])  # dispersal is exponentially distributed
                    disp_probability = scipy.stats.norm.pdf(
                        self._dist_matrix[cell_id], 0, self._disp_rate
                    )
                    disp_vec = disp_probability.flatten()
                    sampling_prob = disp_vec * aval_space

                    selected_cell = self._rs.choice(
                        self._cell_id_n_coord[:, 0],
                        p=sampling_prob / np.sum(sampling_prob),
                    )

                    # keep track of where individuals are being added
                    discrete_dist_per_cell_tmp[selected_cell] += 1

                    # append new individual to current population
                    curr_ind_tmp.append(selected_cell)

                    if len(curr_ind_tmp) >= self._extent:
                        break

                if len(curr_ind_tmp) >= self._extent:
                    break

            rescaled_dist = (
                discrete_dist_per_cell_tmp.reshape(disturbance_matrix.shape)
                / self._disturbance_resolution
            )
            disturbance_matrix = disturbance_matrix + rescaled_dist
            self._curr_ind = self._curr_ind + curr_ind_tmp

        # disturbance_matrix = disturbance_matrix/np.max(disturbance_matrix) * self._max_disturbance
        disturbance_matrix[
            disturbance_matrix > self._max_disturbance
        ] = self._max_disturbance
        self._counter += 1
        if self._mask_disturbance is not None:
            disturbance_matrix = disturbance_matrix * self._mask_disturbance
        return disturbance_matrix


class RoadDisturbanceGenerator(object):
    def __init__(
        self,
        counter,
        peak_disturbance=0.95,
        n_roads_per_step=3,
        scale=0.25,
        step_increase=2,
        seed=1234
    ):
        self._counter = counter
        self._scale = scale
        self._peak_disturbance = peak_disturbance  # set max disturbance
        self._n_roads_per_step = n_roads_per_step
        self._step_increase = step_increase
        self._rs = get_rnd_gen(seed)

    def random_choice_2D(self, length, disturbance_matrix):
        pick_XY = self._rs.choice(
            range(length ** 2),
            p=disturbance_matrix.flatten() / np.sum(disturbance_matrix),
        )
        z = np.zeros(disturbance_matrix.shape).flatten()
        z[pick_XY] = 1
        z = z.reshape(disturbance_matrix.shape)
        return np.where(z == 1)

    def increase_disturbance(self, disturbance_matrix):
        disturbance_matrix *= self._step_increase
        disturbance_matrix[
            disturbance_matrix >= self._peak_disturbance
        ] = self._peak_disturbance
        return disturbance_matrix

    def make_road(self, disturbance_matrix, length, max_disturbance_new_road=0.5):
        indx = np.meshgrid(np.arange(length), np.arange(length))
        rr = self._rs.random()
        horizontal = 0
        if np.sum(disturbance_matrix) == 0 or rr < 0.05:
            pick_X = self._rs.choice([0, length - 1])
            pick_Y = self._rs.choice(range(length))
            horizontal = 1
        elif rr < 0.1:
            pick_X = self._rs.choice(range(length))
            pick_Y = self._rs.choice([0, length - 1])
        else:
            pick_X, pick_Y = self.random_choice_2D(length, disturbance_matrix)
            horizontal = self._rs.choice((0, 2))

        pick_dist = self._rs.randint(int(0.1 * length), length)
        if horizontal:
            locsxy = np.array(
                np.meshgrid(abs(np.arange(pick_X - pick_dist, pick_X + 1)), pick_Y)
            )
            locsxy = locsxy[:, 0, :]
        # print("horizontal", horizontal)
        else:
            locsxy = np.array(
                np.meshgrid(pick_X, abs(np.arange(pick_Y - pick_dist, pick_Y + 1)))
            )
            locsxy = locsxy[:, :, 0]
        # print("vertical", horizontal)

        for i in range(len(locsxy[0, :])):
            # disturbance_matrix += scipy.stats.laplace.pdf(indx[0], loc=locsxy[0, i], scale=sd) * \
            # 			   scipy.stats.laplace.pdf(indx[1], loc=locsxy[1, i], scale=sd)

            new_road = scipy.stats.cauchy.pdf(
                indx[0], loc=locsxy[0, i], scale=self._scale
            ) * scipy.stats.cauchy.pdf(indx[1], loc=locsxy[1, i], scale=self._scale)
            new_road = new_road / np.max(new_road) * max_disturbance_new_road

            disturbance_matrix += new_road

        disturbance_matrix[
            disturbance_matrix >= self._peak_disturbance
        ] = self._peak_disturbance

        return disturbance_matrix

    def updateDisturbance(self, disturbance_matrix):
        length = disturbance_matrix.shape[0]
        if (self._counter + 1) % 2 == 1:
            for i in range(self._n_roads_per_step):
                disturbance_matrix = self.make_road(disturbance_matrix, length)

        else:
            disturbance_matrix = self.increase_disturbance(disturbance_matrix)

        self._counter += 1
        # print(disturbance_matrix)
        return disturbance_matrix

class FixedEmpiricalDisturbanceGenerator(object):
    def __init__(self,
                 counter
                 ):
        self._counter = counter

    def updateDisturbance(self, disturbance_matrix):
        return disturbance_matrix



def get_disturbance(mode, seed=0):
    if mode == -1:
        disturbanceInitializer = TimeIncrementalGaussianDisturbanceGenerator
        distb_obj = disturbanceInitializer(
            0, n_peaks=350, sig=5, peak_disturbance=0, n_peaks_per_step=6
        )
        selectivedistb_obj = disturbanceInitializer(
            0, n_peaks=350, sig=5, peak_disturbance=0, n_peaks_per_step=6
        )
    if mode == 0:
        disturbanceInitializer = LinearIncreaseUniformDisturbanceGenerator
        distb_obj = disturbanceInitializer(counter=0, magnitude=0.2)
        selectivedistb_obj = 0  # i.e. assume = to disturbance_matrix
    elif mode == 1:
        # spatially constrained setting
        disturbanceInitializer = DiffusionDisturbanceGenerator
        distb_obj = disturbanceInitializer(
            counter=0,
            extent=1000,
            n_init_events=1,
            dist_resolution=5,
            expansion_rate=5,
            p_grow_disturbance=1,
            max_disturbance=0.95,
            seed=seed,
        )
        selectivedistb_obj = 0  # i.e. assume = to disturbance_matrix
    elif mode == 2:
        disturbanceInitializer = TimeLinearGaussianDisturbanceGenerator
        distb_obj = disturbanceInitializer(0, 3, 100, mean_disturbance=0.95, n_steps=10, seed=seed)
        selectivedistb_obj = 0  # i.e. assume = to disturbance_matrix

    elif mode == 3:
        disturbanceInitializer = DiffusionDisturbanceGenerator
        distb_obj = disturbanceInitializer(
            counter=0,
            extent=1000,
            n_init_events=1,
            dist_resolution=5,
            expansion_rate=5,
            p_grow_disturbance=1,
            max_disturbance=0.95,
            seed=seed,
        )
        selectivedistb_obj = disturbanceInitializer(
            counter=0,
            extent=2000,
            n_init_events=1,
            dist_resolution=10,
            expansion_rate=5,
            p_grow_disturbance=0.25,
            max_disturbance=0.95,
            seed=seed,
        )
    elif mode == 4:
        disturbanceInitializer = TimeIncrementalGaussianDisturbanceGenerator
        distb_obj = disturbanceInitializer(
            0, n_peaks=350, sig=5, peak_disturbance=0.90, n_peaks_per_step=6, seed=seed
        )
        selectivedistb_obj = 0
    elif mode == 5:
        disturbanceInitializer = DiffusionDisturbanceGenerator
        distb_obj = disturbanceInitializer(
            counter=0,
            extent=1000,
            n_init_events=1,
            dist_resolution=5,
            expansion_rate=5,
            p_grow_disturbance=1,
            max_disturbance=0.95,
            seed=seed,
        )

        # distb_obj = disturbanceInitializer(0, n_peaks=350, sig=5, peak_disturbance=0.95, n_peaks_per_step=8,
        #                                    max_disturbance=0.85, seed=seed)
        disturbanceInitializer = TimeIncrementalGaussianDisturbanceGenerator
        selectivedistb_obj = disturbanceInitializer(
            0,
            n_peaks=350,
            sig=5,
            peak_disturbance=0.95,
            max_disturbance=0.95,
            n_peaks_per_step=8,
            seed=seed,
        )

    elif mode == 6:
        disturbanceInitializer = TimeIncrementalGaussianDisturbanceGenerator
        distb_obj = disturbanceInitializer(
            0, n_peaks=350, sig=5, peak_disturbance=0.90, n_peaks_per_step=4, stop_at=25
        )
        selectivedistb_obj = 0
    elif mode == 7:
        disturbanceInitializer = DiffusionDisturbanceGenerator
        distb_obj = disturbanceInitializer(
            counter=0,
            extent=1000,
            n_init_events=1,
            dist_resolution=5,
            expansion_rate=5,
            p_grow_disturbance=1,
            max_disturbance=0.5,
            seed=seed,
        )
        selectivedistb_obj = disturbanceInitializer(
            counter=0,
            extent=2000,
            n_init_events=1,
            dist_resolution=10,
            expansion_rate=5,
            p_grow_disturbance=0.25,
            max_disturbance=0.95,
            seed=seed,
        )
    elif mode == 8:
        disturbanceInitializer = TimeIncrementalGaussianDisturbanceGenerator
        distb_obj = disturbanceInitializer(
            0,
            n_peaks=350,
            sig=2.5,
            peak_disturbance=0.9,
            max_disturbance=0.9,
            n_peaks_per_step=8,
            seed=seed,
        )

        selectivedistb_obj = disturbanceInitializer(
            0,
            n_peaks=350,
            sig=5,
            peak_disturbance=0.95,
            max_disturbance=0.95,
            n_peaks_per_step=8,
            seed=seed,
        )
    
    # fast-forward TimeIncrementalGaussianDisturbanceGenerator
    elif mode == 9:
        disturbanceInitializer = TimeIncrementalGaussianDisturbanceGenerator
        distb_obj = disturbanceInitializer(
            0,
            n_peaks=350,
            sig=2.5,
            peak_disturbance=0.9,
            max_disturbance=0.9,
            n_peaks_per_step=8,
            seed=seed,
        )
    
        selectivedistb_obj = 0

    return distb_obj, selectivedistb_obj


