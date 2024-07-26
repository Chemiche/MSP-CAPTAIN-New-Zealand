import numpy as np
from scipy import ndimage
# from ..agents.state_monitor import get_quadrant_indx_grid, get_sum_grid_value_quadrant

def _get_quadrant_indx_grid(grid_size, resolution=np.array([1, 1])):
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

def _get_sum_grid_value_quadrant(val, # any 2D array of shape = env.bioDivGrid.h[0].shape
                                 quandrant_grid_indx): # env._quandrant_grid_indx
    return ndimage.sum(val,
                        labels=quandrant_grid_indx[0],
                        index=np.unique(quandrant_grid_indx[0]))


def calc_MSA(A0: np.array, At: np.array):
    # A0 natural state 1D array of species abundances
    # At current state 1D array of species abundances
    A0 = np.round(A0) # round to int to avoid dividing by <1 values
    At = np.round(At)
    if len(A0[A0 > 0]):
        delta = At[A0 > 0] / A0[A0 > 0]
        delta[delta > 1] = 1
        return np.mean(delta)
    else:
        return 0

def calc_MSA_from_grid_resolution(A03d, At3d, resolution=None):
    # A0 natural state 3D array of species abundances (species x lat x lon)
    # At current state 3D array of species abundances (species x lat x lon)
    if resolution is None:
        return calc_MSA(np.einsum('sxy -> s', A03d), np.einsum('sxy -> s', At3d))
    else:
        l = _get_quadrant_indx_grid(A03d.shape[1], np.array([resolution, resolution]))
        msa_granular2c = []
        for i in np.unique(l):
            indx = np.where(l == i)
            if np.sum(A03d[:, indx[0], indx[1]]):
                a0 = np.einsum('sx -> s', A03d[:, indx[0], indx[1]])
                at = np.einsum('sx -> s', At3d[:, indx[0], indx[1]])
                msa_granular2c.append(calc_MSA(a0, at))
        return msa_granular2c


def calc_MSA_from_grid(A03d, At3d, quandrant_grid_indx=None):
    # A0 natural state 3D array of species abundances (species x lat x lon)
    # At current state 3D array of species abundances (species x lat x lon)
    n_species = A03d.shape[0]
    if quandrant_grid_indx is None:
        return calc_MSA(np.einsum('sxy -> s', A03d), np.einsum('sxy -> s', At3d))
    else:
        natural_pop_per_quadrant = np.array([
            _get_sum_grid_value_quadrant(A03d[s], quandrant_grid_indx=quandrant_grid_indx) for s in range(n_species)])

        current_pop_per_quadrant = np.array([
            _get_sum_grid_value_quadrant(At3d[s], quandrant_grid_indx=quandrant_grid_indx) for s in range(n_species)])

        msa_granular2c = [
            calc_MSA(natural_pop_per_quadrant[:,i],
                     current_pop_per_quadrant[:,i]) for i in range(current_pop_per_quadrant.shape[1])]
        return np.array(msa_granular2c)



def calc_PDF_from_grid(natural_ind, new_ind):
    # Potentially Disappeared Fraction
    # https://www.qmul.ac.uk/sbbs/media/sbbs/research/bsc-project/QMUL-QuantifyingBiodiversityImpact2022.pdf
    # env.grid_obj_previous.individualsPerSpecies(), env.grid_obj_most_recent.individualsPerSpecies()
    pdf_metric = - np.mean((new_ind[natural_ind > 0]  - natural_ind[natural_ind > 0]) / natural_ind[natural_ind > 0]) # Eq 3.3, 3.15
    return pdf_metric


def calc_STAR_from_grid(natural_h, current_h, quandrant_grid_indx,
                        sp_natural_range, sp_ext_risk,
                        species_presence_threshold=1,
                        multi=0.29):
    # env.grid_obj_previous.h, env.grid_obj_most_recent.h,
    # env.grid_obj_previous.geoRangePerSpecies(),
    # env.getExtinction_risk_labels(), env._quandrant_grid_indx
    n_species = natural_h.shape[0]
    # http://opus.sanbi.org/jspui/bitstream/20.500.12143/7548/1/s41559-021-01432-0.pdf
    # H_i: restorable_area_i
    AOH = sp_natural_range
    # LC: 0; Near Threatened = 1; Vulnerable = 2; Endangered = 3; Critically Endangered = 4
    W_s = 4 - sp_ext_risk

    # natural range (AOH) per quadrant
    sp_natural_range_in_quadrant = np.array([_get_sum_grid_value_quadrant(natural_h[i] > species_presence_threshold,
                                                                          quandrant_grid_indx=quandrant_grid_indx) for i in range(n_species)])
    sp_range_in_quadrant = np.array([_get_sum_grid_value_quadrant(current_h[i] > species_presence_threshold,
                                                                  quandrant_grid_indx=quandrant_grid_indx) for i in range(n_species)])
    # percentage of suitable range in quadrant out of total
    P_si = np.einsum('sq, s -> sq', sp_natural_range_in_quadrant, 1 / AOH)
    STAR_t = np.einsum('sq, s -> q', P_si, W_s) # one value per quadrant

    # restorable range per quadrant
    res_rng = sp_natural_range_in_quadrant - sp_range_in_quadrant
    # percentage of restorable range
    H_si = np.einsum('sq, s -> sq', res_rng, 1 / AOH)
    STAR_r = np.einsum('sq, s -> q', H_si, W_s * multi)
    return STAR_t, STAR_r