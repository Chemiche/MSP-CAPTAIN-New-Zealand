import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os
import pandas as pd
import scipy.ndimage
import seaborn as sns
import h5py
import geopandas as gpd
import geopandas as gpd
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import rioxarray as rxr
import rasterio
import earthpy as et
from shapely.geometry import Polygon, shape, Point
from shapely.ops import cascaded_union, unary_union
from rasterio.enums import Resampling
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
import captain as cn
import glob
import matplotlib.backends.backend_pdf
from .misc import get_rnd_gen, print_update
import geopy
import geopy.distance
import scipy.stats

def load_sdms(f_names, species_cutoff=None,rnd_seed=None):
    d, n, c = [], [], []
    n_species = 0
    for f_name in f_names:
        print("Loading:", f_name)
        with h5py.File(f_name, 'r') as f:
            d.append(f.get("density")[()])
            n.append(f.get("names")[()])
            c.append(f.get("count")[()])
            n_species += len(f.get("names")[()])
    da = np.array(d)
    species_names_list = np.array(n).flatten().astype(str)

    density = da.reshape(n_species, da.shape[2], da.shape[3])
    density[np.isnan(density)] = 0

    occ_count = np.array(c).flatten()

    if rnd_seed is not None:
        # randomize order
        rs = get_rnd_gen(rnd_seed)
        indx = rs.choice(range(n_species), n_species, replace=False)
        density = density[indx, :, :]
        species_names_list = species_names_list[indx]
        occ_count = occ_count[indx]

    if species_cutoff is not None:
        density = density[:species_cutoff, :, :] + 0
        species_names_list = species_names_list[:species_cutoff]
        n_species = species_cutoff
        occ_count = occ_count[:species_cutoff]

    return density, n_species, species_names_list, occ_count


def load_wwf(f_name,
             countries,
             lat_point_list,
             lon_point_list,
             plot_file=None,
             resolution=(-0.25, 0.25)):
    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    polygons = [world[world.name == c].geometry for c in countries]
    geo_boundary = gpd.GeoSeries(unary_union(polygons))
    gdf_boundary = gpd.GeoDataFrame(geometry=geo_boundary, crs='epsg:4326')
    captain_area = gdf_boundary.overlay(polygon, how='intersection')

    # WWF BIOMES
    wwf = gpd.read_file(f_name)
    _ = wwf.overlay(captain_area, how='intersection').plot(column='BIOME',
                                                            cmap='YlGn_r',
                                                            linewidth=0,
                                                            legend=True,
                                                            categorical=True,
                                                            legend_kwds={'loc': 'center left',
                                                                         'bbox_to_anchor': (1, 0.5),
                                                                         'fmt': "{:.0f}"
                                                                         }
                                                            )

    if plot_file is not None:
        plt.gca().set_title("WWF Biomes", fontweight="bold", fontsize=15)
        plt.savefig(plot_file)

    # labels = ['1: Rain forest', '2: Dry forest', '7: Tropical grassland',
    #           '8: Temperate grassland', '9: Flooded grassland',
    #           '13: Xeric grassland', '14: Mangroves']

    # rasterize
    wwf_captain = wwf.overlay(captain_area, how='intersection')
    wwf_captain_raster = make_geocube(vector_data=wwf_captain,
                                      measurements=["BIOME"],
                                      resolution=resolution,
                                      )
    wwf_array = np.array(wwf_captain_raster['BIOME'].values)
    mask_suitability = (np.isfinite(wwf_array)).astype(int)  # remove occs in the sea
    return wwf, wwf_array, captain_area, mask_suitability


def get_habitat_suitability(h_tmp, lower=0.25, integer=True):
    h = h_tmp + 0
    h[np.isnan(h)] = 0
    h[h < lower] = 0
    if integer:
        h[h > 0] = 1
    return h


def plot_map(m, z=None, cmap=None, title=None, nan_to_zero=False,
             fig_s=[6.5, 5.5], show=False, outfile=None, dpi=250,
             vmin=None, vmax=None):
    fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
    fig.tight_layout()
    m_plotted = (m + 0).astype(float)
    if nan_to_zero:
        m_plotted[np.isnan(m_plotted)] = 0

    if z is not None:
        m_plotted[np.isnan(z)] = np.float64(np.nan)

    sns.heatmap(m_plotted, cmap=cmap,
                xticklabels=False,
                yticklabels=False,
                vmin=vmin, vmax=vmax)
    if title is not None:
        plt.gca().set_title(title, fontweight="bold", fontsize=15)
    if show:
        fig.show()
    elif outfile is not None:
        plt.savefig(outfile, dpi=dpi)
        plt.close()
    else:
        return fig

def load_land_use(land_use_raster):
    img = tiff.imread(land_use_raster)
    return np.array(img)

def load_pop_density(population_file):
    img = tiff.imread(population_file)
    population = np.array(img)
    population[population < 0.001] = 0

    population_discrete = population * 0
    population_discrete[population > 5] = 0.1
    population_discrete[population > 10] = 0.3
    population_discrete[population > 50] = 0.75
    population_discrete[population > 200] = 0.95

    return population, population_discrete

def get_combined_disturbance(population_discrete, land_use, mask_suitability):
    combined_disturbance = population_discrete + 0
    combined_disturbance[land_use == 200] = 0  # np.nan # open sea
    combined_disturbance[land_use == 40] = 0.75  # cultivated
    combined_disturbance[land_use == 50] = 0.99  # urban
    # set marine areas to 0 cost
    combined_disturbance = combined_disturbance * mask_suitability
    return combined_disturbance


def get_cost_layer(land_use, population, mask_suitability, max_cost = 100,
                   baseline_per_person=1):
    cost_layer = np.ones(land_use.shape)
    # baseline costs based on land use
    cost_layer[land_use == 50] = 100  # urban (added value per person)
    cost_layer[land_use == 40] = 10  # cultivated
    cost_layer[land_use == 200] = 0  # np.nan # open sea
    cost_layer = cost_layer + baseline_per_person * population
    cost_layer[cost_layer > max_cost] = max_cost  # truncate cost to budget
    # set marine areas to 0 cost
    cost_layer = cost_layer * mask_suitability
    cost_layer = cost_layer / np.max(cost_layer)
    return cost_layer


def generate_h3d(density, suitability, mask_suitability, max_K_cell, K_biome, max_K_multiplier,
                 K_as_f_diversity=False):
    h3d = np.einsum('sxy, sxy, xy -> sxy', density, suitability, mask_suitability)
    # biome carrying capacity
    max_K_biome = max_K_cell * K_biome * max_K_multiplier
    max_K_biome[max_K_biome == 0] = 1
    ind_tmp = np.einsum('sxy -> xy', h3d)
    den = ind_tmp / max_K_biome
    den[den == 0] = 1
    h3d = np.einsum('sxy, xy -> sxy', h3d, 1 / den)
    # set empirical carrying capacity
    K_cells = np.einsum('sxy -> xy', h3d)
    # set natural carrying capacity to empirical one times a factor
    # if factor == 1: will lose diversity because mortality is high at carrying capacity
    # could multiply h3d by a value to have "individuals" instead of presence/absence
    K_max = K_cells + 0
    # set non-marine areas to a minimum carrying capacity
    K_max[K_cells == 0] = mask_suitability[K_cells == 0]
    K_max = K_max * mask_suitability
    return h3d, K_max, K_cells


def load_redlist_data(rl_file, species_names, allow_NAs=False):
    rl_tbl = pd.read_csv(rl_file, sep='\t')
    sp_rl = rl_tbl['species'].to_numpy().astype(str)
    status_rl = rl_tbl['status'].to_numpy().astype(str)
    status = []
    not_found = []
    for n in range(len(species_names)):
        if species_names[n].replace(' ', '_') in sp_rl:
            status_rl_n = status_rl[np.where(sp_rl == species_names[n].replace(' ', '_'))[0][0]]
            if status_rl_n in ['CR', 'EN', 'VU', 'NT', 'LC']:
                status.append(status_rl_n)
            else:
                if allow_NAs:
                    status.append('DD')
                else:
                    status.append(np.random.choice(['CR', 'EN', 'VU', 'NT', 'LC']))
                not_found.append(species_names[n])
        else:
            if allow_NAs:
                status.append('DD')
            else:
                status.append(np.random.choice(['CR', 'EN', 'VU', 'NT', 'LC']))
            not_found.append(species_names[n])

    status = np.array(status)
    status[status == 'CR'] = 0
    status[status == 'EN'] = 1
    status[status == 'VU'] = 2
    status[status == 'NT'] = 3
    status[status == 'LC'] = 4
    if allow_NAs:
        status[status == 'DD'] = np.nan
    else:
        status = status.astype(int)
    return status, not_found

def plot_restoration_priority_map(out_files, wwf, captain_area,
                                  resolution=0.25, plot_file=None,
                                  jitter=0, cmap='viridis',
                                  vmin=None, vmax=None,
                                  background_color='#3E0851',
                                  markersize=5):
    x = []
    for f in out_files:
        x.append(np.load(f))

    priority = np.mean(x, axis=0)

    wwf_captain = wwf.overlay(captain_area, how='intersection')
    wwf_captain_raster = make_geocube(vector_data=wwf_captain,
                                      measurements=["BIOME"],
                                      resolution=(-resolution, resolution),
                                      )
    xy = np.meshgrid(wwf_captain_raster.x.values, wwf_captain_raster.y.values)

    priority[priority == 0] = np.nan
    p = {'predicted': priority.flatten(),
         'x': xy[0].flatten(),
         'y': xy[1].flatten()
         }

    p_pd = pd.DataFrame(p)

    # jitter
    if jitter:
        p_pd['x'] += np.random.uniform(0, jitter, len(p_pd['x']))
        p_pd['y'] += np.random.uniform(0, jitter, len(p_pd['y']))


    out_gdf = gpd.GeoDataFrame(p_pd['predicted'],
                               geometry=gpd.points_from_xy(p_pd['x'], p_pd['y']),
                               crs='epsg:4326')


    biomes_gdf = wwf.overlay(captain_area, how='intersection')

    ax = biomes_gdf.plot(color=background_color,
                         # column='BIOME',
                         # cmap='YlGn_r',
                         # alpha=0.6,
                         linewidth=0,
                         # legend=True,
                         categorical=True,
                         legend_kwds={'loc': 'center left',
                                      'bbox_to_anchor': (1, 0.5),
                                      'fmt': "{:.0f}"
                                      }
                         )

    out_gdf.plot(column="predicted", ax=ax,
                 markersize=markersize, cmap=cmap,
                 vmin=vmin, vmax=vmax,
                 legend=True)
    plt.gca().set_title("Restoration priorities", fontweight="bold", fontsize=15)
    if plot_file is not None:
        plt.savefig(plot_file)
        print("Plot saved as:", plot_file)
    else:
        plt.show()


### NZ functions
def calc_all_to_all_geo_distance(lat, lon):
    # print("calculating distances...")
    coord_size = len(lon)
    dist = np.zeros(tuple(np.repeat(coord_size, 4)))
    for ii, i in enumerate(lat):
        for jj, j in enumerate(lon):
            for nn, n in enumerate(lat):
                for mm, m in enumerate(lon):
                    # relative dispersal probability: always 1 at distance = 0
                    # the actual number of offspring is modulated by growth_rate
                    dist[ii, jj, nn, mm] = geopy.distance.geodesic((i, j), (n, m)).km
    return dist


def calc_all_to_all_geo_distanc_grid(lat2d, lon2d):

    if lon2d.shape[0] !=lon2d.shape[1]:
        print("Error in calc_all_to_all_geo_distance: lon2d.shape[0] != lon2d.shape[1]")
    # print("calculating distances...")
    coord_size = lon2d.shape[0] # assumning square shape
    dist = np.zeros(tuple(np.repeat(coord_size, 4)))

    for ii in range(coord_size):
        print_update("done: %s / %s " % (ii, coord_size))
        for jj in range(coord_size):
            for nn in range(coord_size):
                for mm in range(coord_size):
                    from_loc  = (lat2d[ii, jj], lon2d[ii, jj])
                    to_loc = (lat2d[nn, mm], lon2d[nn, mm])
                    dist[ii, jj, nn, mm] = geopy.distance.geodesic(from_loc, to_loc).km

    return dist





def get_dispersal_prob(distances, lambda0, max_dist=None):
    r = 1 / lambda0
    dispersal = np.exp(r * distances)
    if max_dist is not None:
        dispersal[distances > max_dist] = 0
    return dispersal

def parse_tif_file(tif_file, rescale_factor=1., min_cutoff=np.nan):
    sdm = rasterio.open(tif_file)
    dd = np.squeeze(sdm.read(out_shape=(sdm.count,
                                        int(sdm.height * rescale_factor),
                                        int(sdm.width * rescale_factor))))
    if min_cutoff is not np.nan:
        dd[dd < min_cutoff] = 0
    return sdm, dd


def calc_dispersal_prob_from_grid(coords=None, length_x=None, length_y=None, lambda0=1, max_disp=1, kernel = "normal"):
    if coords is None and length_x is not None:
        coords = np.meshgrid(np.arange(length_y), np.arange(length_x))
    elif coords is not None:
        length_x = coords[0].shape[0]
        length_y = coords[0].shape[1]
        # coords = np.meshgrid(coords[1], coords[0])
    else:
        print("Error: must specify x or len(x)")
    disp = np.zeros((length_x, length_y, length_x, length_y))
    for x in range(length_x):
        for y in range(length_y):
            if kernel == "normal":
                dist_x = scipy.stats.norm.pdf(coords[1], loc=coords[1][x, y], scale=lambda0)
                dist_y = scipy.stats.norm.pdf(coords[0], loc=coords[0][x, y], scale=lambda0)
            if kernel == "laplace":
                dist_x = scipy.stats.laplace.pdf(coords[1], loc=coords[1][x, y], scale=lambda0)
                dist_y = scipy.stats.laplace.pdf(coords[0], loc=coords[0][x, y], scale=lambda0)
            disp[x, y] = dist_x * dist_y

    # normalize (max dispersal at 0 distance = 1
    disp /= np.max(disp)
    return disp * max_disp



from numba import jit
import random
import scipy.stats

small_number = 1e-10


@jit(nopython=True)
def dispersalDistancesRectangle(length_x, length_y, lambda_0):
    # print("calculating distances...")
    dumping_dist = np.zeros((length_x, length_y, length_x, length_y))
    for i in range(0, length_x):
        # print("Calculating distances: %s / %s" % (i, length_x))
        for j in range(0, length_y):
            for n in range(0, length_x):
                for m in range(0, length_y):
                    exp_rate = 1.0 / lambda_0
                    # relative dispersal probability: always 1 at distance = 0
                    # the actual number of offspring is modulated by growth_rate
                    dumping_dist[i, j, n, m] = np.exp(
                        -exp_rate * np.sqrt((i - n) ** 2 + (j - m) ** 2)
                    )
    return dumping_dist




def load_clip_resize_tif(tif_file,
                         polygon=None, # [minx, miny, maxx, maxy]
                         crs="EPSG:4326",
                         scale_factor=1.,
                         size=None,
                         return_coord_df=False
                         ):
    ch_bio = rxr.open_rasterio(tif_file, masked=True)
    if (polygon is None):
        ch_bio_cut = ch_bio
    else:
        # ch_bio_cut = ch_bio.rio.clip_box(minx=clip[0],
        #                                  miny=clip[1],
        #                                  maxx=clip[2],
        #                                  maxy=clip[3],
        #                                  crs=crs)
        ch_bio = ch_bio.rio.reproject(polygon.crs)
        ch_bio_cut = ch_bio.rio.clip(polygon['geometry'], polygon.crs, drop=False, invert=True)

    # downsample by average
    if size is not None:
        new_width = size[0]
        new_height = size[1]
    else:
        new_width = int(ch_bio_cut.rio.width * scale_factor)
        new_height = int(ch_bio_cut.rio.height * scale_factor)

    xds_down_sampled = ch_bio_cut.rio.reproject(
        ch_bio_cut.rio.crs,
        shape=(new_height, new_width),
        resampling=Resampling.average,
    )

    if return_coord_df:
        xds_down_sampled.name = "temp"
        ch_df  = xds_down_sampled.to_dataframe().reset_index()
        ch_gdf = gpd.GeoDataFrame(ch_df, geometry=gpd.points_from_xy(ch_df.x, ch_df.y), crs='epsg:4326')
        ch_gdf['temp_scaled'] =  ch_gdf["temp"]*ch_bio_cut.attrs['scale_factor'] + ch_bio_cut.attrs['add_offset']
        ch_gdf = ch_gdf[['x', 'y', 'geometry', 'temp_scaled']]
        return ch_gdf
    else:
        temp_clipped = xds_down_sampled.to_numpy().squeeze()
        return temp_clipped



def load_sdms_from_dir(wd,
                       clip=None,
                       size=None, # [minx, miny, maxx, maxy]
                       rescale_factor=0.05, cutoff=0.05, plot=False):
    sp_dirs = glob.glob(os.path.join(wd, "*"))
    sdm_list = []
    for sp_dir in sp_dirs:
        if 1: #try:
            tif_file = glob.glob(os.path.join(sp_dir, "*.tif"))[0]
            print(tif_file)
            scaled_dd = load_clip_resize_tif(tif_file,
                                                  scale_factor=rescale_factor,
                                                  clip=clip, size=size
                                                  )
            if cutoff is not np.nan:
                scaled_dd[scaled_dd < cutoff] = 0
            if plot:
                sns.heatmap(scaled_dd, cmap="YlGnBu")
                plt.title(os.path.basename(sp_dir))
                plt.show()
            sdm_list.append(scaled_dd)
        # except:
        #     print("TIF not found in", sp_dir)
    return np.array(sdm_list)




























