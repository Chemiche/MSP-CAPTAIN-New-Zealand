import os
import numpy as np
from scipy import ndimage
import glob
np.set_printoptions(suppress=True, precision=3)  # prints floats, no scientific notation
from ..agents.state_monitor import get_quadrant_indx_grid, get_sum_grid_value_quadrant

def z_transform(x):
 return (x - np.nanmean(x)) / np.nanstd(x)

def feature_transform(x, delta, denom):
    return (x - delta) / denom

def get_features_sdm(bathy, temp, oxygen,
                     future_temp=None,
                     future_oxygen=None,
                     include_coords=None, # xarray
                     reorder=None,
                     convolution_padding=10,
                     rescalers=None,
                     ):
    b = bathy.flatten()
    t = temp.flatten()
    o = oxygen.flatten()

    if reorder is None:
        reorder = np.arange(len(b))

    feat_list = [feature_transform(b, rescalers['bathy'][0], rescalers['bathy'][1])[reorder],
                 feature_transform(t, rescalers['temp'][0], rescalers['temp'][1])[reorder],
                 feature_transform(o, rescalers['oxygen'][0], rescalers['oxygen'][1])[reorder]]

    if future_temp is not None:
        t_fut = future_temp.flatten()
        o_fut = future_oxygen.flatten()
        feat_list_future = [feature_transform(b, rescalers['bathy'][0], rescalers['bathy'][1])[reorder],
                            feature_transform(t_fut, rescalers['temp'][0], rescalers['temp'][1])[reorder],
                            feature_transform(o_fut, rescalers['oxygen'][0], rescalers['oxygen'][1])[reorder]]

    if include_coords is not None:
        coords = np.meshgrid(include_coords[0]['x'].to_numpy(),
                               include_coords[0]['y'].to_numpy())
        l = coords[0].flatten()
        g = coords[1].flatten()

        feat_list = feat_list + [feature_transform(l, rescalers['lat'][0], rescalers['lat'][1])[reorder],
                                 feature_transform(g, rescalers['lon'][0], rescalers['lon'][1])[reorder]]
        if future_temp is not None:
            feat_list_future = feat_list_future + [
                feature_transform(l, rescalers['lat'][0], rescalers['lat'][1])[reorder],
                feature_transform(g, rescalers['lon'][0], rescalers['lon'][1])[reorder]]

    if convolution_padding > 1:
        k = np.ones((convolution_padding, convolution_padding)) / (convolution_padding ** 2)
        temp_tmp = temp[0].to_numpy() + 0
        temp_tmp[np.isfinite(temp_tmp) == False] = -10
        temp_conv = ndimage.convolve(temp_tmp, k, mode='reflect')
        temp_conv_z = feature_transform(temp_conv.flatten(), rescalers['temp'][0], rescalers['temp'][1])

        bathy_tmp = bathy[0].to_numpy() + 0
        bathy_conv = ndimage.convolve(bathy_tmp, k, mode='reflect')
        bathy_conv_z = feature_transform(bathy_conv.flatten(), rescalers['bathy'][0], rescalers['bathy'][1])
        feat_list = feat_list + [temp_conv_z[reorder],
                                 bathy_conv_z[reorder]]

        if future_temp is not None:
            f_temp_tmp = future_temp[0].to_numpy() + 0
            f_temp_tmp[np.isfinite(temp_tmp) == False] = -10
            f_temp_conv = ndimage.convolve(temp_tmp, k, mode='reflect')
            f_temp_conv = feature_transform(f_temp_conv.flatten(), rescalers['temp'][0], rescalers['temp'][1])
            feat_list_future = feat_list_future + [f_temp_conv[reorder],
                                                   bathy_conv_z[reorder]]


    features = np.array(feat_list).T
    if future_temp is not None:
        features_future = np.array(feat_list_future).T
    else:
        features_future = None
    return features, features_future


def get_sdm_data(sdm_files, taxon_index, cropped=None):
    import rioxarray as rxr
    sp_data = rxr.open_rasterio(sdm_files[taxon_index], masked=True)
    if cropped is not None:
        sp_data_cropped = crop_data(sp_data, cropped)
    else:
        sp_data_cropped = sp_data.to_numpy()[0]
    taxon_name = os.path.basename(sdm_files[taxon_index]).split(".tif")[0]
    return sp_data_cropped, taxon_name

def crop_data(layer, cropped):
    layer_np = layer.to_numpy()[0][:, cropped]
    return layer_np


def get_data_from_list(sdm_dir, tag="/*.tif", max_species_cuptoff=None, rescale=False):
    sdm_files = np.sort(glob.glob(sdm_dir + tag))
    sdms = []
    sp_names = []
    for i in range(len(sdm_files)):
        sp_data, taxon_name = get_sdm_data(sdm_files, i, cropped=None)
        if rescale:
            sp_data /= np.nanmax(sp_data)
        sdms.append(sp_data)
        sp_names.append(taxon_name)
        if i + 1 == max_species_cuptoff:
            break

    sdms = np.squeeze(np.array(sdms))  # shape = (species, lon, lat)
    return sdms, sp_names



def grid_to_graph(grid, reference_grid_pu, n_pus=None, nan_to_zero=False):
    if n_pus is None:
        n_pus = reference_grid_pu[reference_grid_pu > 0].size
    grid_length = np.round(np.sqrt(n_pus)).astype(int) + 1
    if grid_length % 2 != 0: # make it a multiple of 2
        grid_length += 1

    if len(grid.shape) == 3:
        m_graph = []
        for scaled_dd in grid:
            prep_vec = np.zeros(grid_length ** 2)
            species_vec = scaled_dd[reference_grid_pu > 0].flatten()
            prep_vec[:n_pus] += species_vec
            species_grid = prep_vec.reshape((grid_length, grid_length))
            species_grid[np.isnan(species_grid)] = 0
            m_graph.append(species_grid)

        m_graph = np.array(m_graph)

    elif len(grid.shape) == 2:
        m_graph = np.zeros(grid_length ** 2)
        m_graph[:n_pus] += grid[reference_grid_pu > 0].flatten()
        m_graph = m_graph.reshape((grid_length, grid_length))

    else:
        m_graph = None
        print("ERROR: Data not recognized")

    if nan_to_zero:
        m_graph[np.isnan(m_graph)] = 0

    return m_graph, n_pus, grid_length

def graph_to_grid(graph, reference_grid_pu, n_pus=None, nan_to_zero=False):
    if n_pus is None:
        n_pus = reference_grid_pu[reference_grid_pu > 0].size

    original_grid_shape = reference_grid_pu.shape

    if len(graph.shape) == 3:
        m_grid = []
        for var in graph:
            tmp = var.flatten()[:-(var.size - n_pus)] + 0
            z = np.zeros(original_grid_shape)
            z[reference_grid_pu > 0] += tmp
            if nan_to_zero:
                z[reference_grid_pu == 0] = np.nan
            m_grid.append(z)

        m_grid = np.array(m_grid)
    else:
        tmp = graph.flatten()[:-(graph.size - n_pus)] + 0
        m_grid = np.zeros(original_grid_shape)
        m_grid[reference_grid_pu > 0] += tmp
        if nan_to_zero:
            m_grid[reference_grid_pu == 0] = np.nan

    return m_grid\




























