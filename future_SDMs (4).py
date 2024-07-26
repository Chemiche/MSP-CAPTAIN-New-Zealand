import os
import glob
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True, precision=3)
import rasterio
import captain as cn
import captain
import seaborn as sns
import matplotlib.pyplot as plt
from captain.utilities import empirical_data_parser as cn_util
from captain.utilities import sdm_utils as sdm_utils
# from captain.utilities import tf_nn as cn_nn
from shapely.geometry import Polygon, shape, Point
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
import geopandas as gpd
import pyproj
import scipy.ndimage



# data_wd = r"C:\Users\mkgue\Downloads\NewZealand_MSP-main (3)" # change to your path!
# data_wd = r"C:\Users\mkgue\Downloads\NewZealand_MSP-main (3)\NewZealand_MSP-main\sdm_predict"
data_wd = "/Users/dsilvestro/Software/CAPTAIN/NewZealand_MSP/sdm_predict"
env_data_wd = os.path.join(data_wd, "env_layers")
sdm_data_wd = os.path.join(data_wd, "present_sdms")
plot_layers = False
log_bathymetry = True
truncate_bathymetry = 0
rm_duplicates = 0
truncate_layers = True # Truncate max bathymetry and min temp to observed range


# load env predictors
bathy = rxr.open_rasterio(os.path.join(data_wd, env_data_wd, "Bathymetry_Current.tif"), masked=True)
temp = rxr.open_rasterio(os.path.join(data_wd, env_data_wd, "Temperature_Current.tif"), masked=True)
future_temp = rxr.open_rasterio(os.path.join(data_wd, env_data_wd, "Temperature_Future.tif"), masked=True)
oxygen = rxr.open_rasterio(os.path.join(data_wd, env_data_wd, "Oxygen_Current.tif"), masked=True)
future_oxygen = rxr.open_rasterio(os.path.join(data_wd, env_data_wd, "Oxygen_Future.tif"), masked=True)

# crop data based on bathymtetry and oxygen layers
b = bathy[0].to_numpy()
cropped_b = np.isfinite(b[0]).astype(int)
o = oxygen.to_numpy()[0]
cropped_o = np.isfinite(o[0]).astype(int)
cropped = np.where(cropped_b * cropped_o == 1)[0]


# prep env layers
bathy_l = sdm_utils.crop_data(bathy, cropped)
bathy_l[bathy_l > 0] = np.nan
if truncate_bathymetry > 0:
    bathy_l[bathy_l < -truncate_bathymetry] = -truncate_bathymetry
# log transf
if log_bathymetry:
    bathy_l[np.isfinite(bathy_l)] = np.log(np.abs(bathy_l[np.isfinite(bathy_l)]))

temp_l = sdm_utils.crop_data(temp, cropped)
future_temp_l = sdm_utils.crop_data(future_temp, cropped)
oxygen_l = sdm_utils.crop_data(oxygen, cropped)
future_oxygen_l = sdm_utils.crop_data(future_oxygen, cropped)


# load SDM data
sdm_files = np.sort(glob.glob(os.path.join(sdm_data_wd, "*")))

if plot_layers:
    sdm_tmp, taxon_name = sdm_utils.get_sdm_data(sdm_files, 0, cropped)
    # sdm_tmp[sdm_tmp < 0.5] = 0
    sns.heatmap(sdm_tmp, cmap="viridis")
    plt.title(taxon_name)
    plt.show()

    # bathy_l_tmp = bathy_l + 0
    # bathy_l_tmp[bathy_l_tmp > 0] = np.nan
    sns.heatmap(bathy_l, cmap="Blues")
    plt.title("Bathymetry (log)")
    plt.show()

    sns.heatmap(temp_l, cmap="Reds", vmin=0, vmax=20)
    plt.title("Temp")
    plt.show()

    sns.heatmap(future_temp_l, cmap="Reds")
    plt.title("Future temp")
    plt.show()

    sns.heatmap(future_temp_l - temp_l, cmap="bwr",vmax=3, vmin=-3)
    plt.title("Temp anomaly")
    plt.show()

    sns.heatmap(oxygen_l, cmap="BuGn")
    plt.title("Oxygen")
    plt.show()

    sns.heatmap(future_oxygen_l, cmap="BuGn")
    plt.title("Future oxygen")
    plt.show()








# build training features
rs = cn.get_rnd_gen(1234)
b = bathy_l.flatten()
rnd_order = rs.choice(np.arange(len(b)), size=len(b), replace=False)

rescalers = {
    'bathy': [np.nanmean(bathy_l), np.nanstd(bathy_l)],
    'temp': [np.nanmean(temp_l), np.nanstd(temp_l)],
    'oxygen': [np.nanmean(oxygen_l), np.nanstd(oxygen_l)]
}

features_training, _ = sdm_utils.get_features_sdm(bathy=bathy_l,
                                                  temp=temp_l,
                                                  oxygen=oxygen_l,
                                                  reorder=rnd_order,
                                                  include_coords=None,
                                                  convolution_padding=0,
                                                  rescalers=rescalers)





# build and train model for species 'taxon_ind'
plot_sdm = True
show = False # save plots to files instead
taxon_ind = 0

true_sdm = []
predicted_present_sdm = []
predicted_future_sdm = []
species_names = []

while True:

    try:
        sp_data, taxon_name = sdm_utils.get_sdm_data(sdm_files, taxon_ind, cropped)
    except(IndexError):
        break

    labels = sp_data.flatten()[rnd_order]
    print("Running taxon:", taxon_name)
    # features_training[np.isfinite(features_training) == False] = -1


    if truncate_layers:
        features_training[np.isnan(labels) == True, 0] = np.nanmax(features_training[np.isfinite(labels) == True, 0])  # bathy
        features_training[np.isnan(labels) == True, 1] = np.nanmin(features_training[np.isfinite(labels) == True, 1])  # temp
        # features_training[np.isnan(labels) == True, 2] = np.nanmean(features_training[np.isnan(labels) == True, 2])  # oxigen


    labels_cp = labels + 0
    # labels[np.isfinite(labels) == False] = 0
    # subset to cells with SDM estimates
    labels_rd = labels[np.isfinite(labels) == True]
    features_training_rd = features_training[np.where(np.isfinite(labels) == True)[0],:]

    # remove NAs
    labels_rd = labels_rd[np.isfinite(np.sum(features_training_rd,1)) == True]
    features_training_rd = features_training_rd[np.isfinite(np.sum(features_training_rd,1)) == True]


    # remove duplicates
    if rm_duplicates == 0:
        unique_indx = np.arange(len(labels_rd))
    else:
        p = features_training_rd[:,0] * (features_training_rd[:,1] + 10) * labels_rd + rnd
        u, unique_indx = np.unique(np.round(p, 6), return_index=True)
        
    model = cn.build_nn(features_training_rd,
                        dense_nodes=[128, 64],
                        dropout_rate=0,
                        loss_f='mae')

    h = cn.fit_nn(features_training_rd[unique_indx],
                  labels_rd[unique_indx],
                  model,
                  patience=5,
                  verbose=1,
                  batch_size=100,
                  max_epochs=100)



    # re-predict the present SDM and predict future
    features_present, features_future = sdm_utils.get_features_sdm(bathy=bathy_l,
                                                                   temp=temp_l,
                                                                   oxygen=oxygen_l,
                                                                   future_temp=future_temp_l,
                                                                   future_oxygen=future_oxygen_l,
                                                                   reorder=None,
                                                                   include_coords=None,
                                                                   convolution_padding=0,
                                                                   rescalers=rescalers)

    if truncate_layers:
        labels = sp_data.flatten()
        features_present[np.isnan(labels) == True, 0] = np.nanmax(features_present[np.isfinite(labels) == True, 0])  # bathy
        features_present[np.isnan(labels) == True, 1] = np.nanmin(features_present[np.isfinite(labels) == True, 1])  # temp
        # features_training[np.isnan(labels) == True, 2] = np.nanmean(features_training[np.isnan(labels) == True, 2])  # oxigen

        features_future[np.isnan(labels) == True, 0] = np.nanmax(features_future[np.isfinite(labels) == True, 0])  # bathy
        features_future[np.isnan(labels) == True, 1] = np.nanmin(features_future[np.isfinite(labels) == True, 1])  # temp
        # features_training[np.isnan(labels) == True, 2] = np.nanmean(features_training[np.isnan(labels) == True, 2])  # oxigen

        
    y = model.predict(features_present)
    pred_present = y.reshape(sp_data.shape)

    y = model.predict(features_future)
    pred_future = y.reshape(sp_data.shape)
    
    if plot_sdm:
        
        try:
            os.mkdir(os.path.join(data_wd, "sdm_plots"))
        except FileExistsError:
            pass
        
        fig = plt.figure(figsize=(12, 4.5), layout="constrained")
        ax1 = fig.add_subplot(131)
        # plot true present SDM
        sns.heatmap(sp_data, cmap="viridis", xticklabels=False, yticklabels=False)
        plt.title("%s True SDM" % taxon_name)

        ax2 = fig.add_subplot(132)
        # features_present
        pred = pred_present + 0
        pred[np.isnan(sp_data)] = np.nan
        print("MAE:", np.nanmean(np.abs(pred - sp_data)))
        sns.heatmap(pred, cmap="viridis", xticklabels=False, yticklabels=False)
        plt.title("%s Predicted SDM - Present" % taxon_name)

        ax3 = fig.add_subplot(133)
        # predict and plot future SDM
        pred = pred_future + 0
        pred[np.isnan(sp_data)] = np.nan
        sns.heatmap(pred, cmap="viridis", xticklabels=False, yticklabels=False)
        plt.title("%s Predicted SDM - Future" % taxon_name)
        
        #-
        # plt.scatter(sp_data.flatten(), pred_present.flatten(), alpha=0.01)
        # plt.show()
        #- 
        if show:
            fig.show()
        else:
            file_name = os.path.join(data_wd, "sdm_plots", taxon_name + ".png")
            plt.savefig(file_name, dpi=300)
            plt.close()
            print("Plot saved as:", file_name)
        
    true_sdm.append(sp_data)
    predicted_present_sdm.append(pred_present)
    predicted_future_sdm.append(pred_future)
    species_names.append(taxon_name)

    taxon_ind += 1


np.save(os.path.join(data_wd, "true_sdms.npy"), np.array(true_sdm))
np.save(os.path.join(data_wd, "present_sdms.npy"), np.array(predicted_present_sdm))
np.save(os.path.join(data_wd, "future_sdms.npy"), np.array(predicted_future_sdm))
np.save(os.path.join(data_wd, "species_names.npy"), np.array(species_names))


# try if it worked
true_sdm = np.load(os.path.join(data_wd, "true_sdms.npy"))
predicted_present_sdm = np.load(os.path.join(data_wd, "present_sdms.npy"))
predicted_future_sdm = np.load(os.path.join(data_wd, "future_sdms.npy"))
species_names = np.load(os.path.join(data_wd, "species_names.npy"))


summary_res = []

# calculate stats
for species_indx in range(len(species_names)):
    s1 = predicted_present_sdm[species_indx]
    s2 = predicted_future_sdm[species_indx]

    # we could also filter out all cells that were not in the original map:
    crop_to_original = True
    if crop_to_original:
        s0 = true_sdm[species_indx]
        s1[np.isnan(s0)] = np.nan
        s2[np.isnan(s0)] = np.nan

    # calculate suitability change (total suitability)
    percentage_change = 100 * (np.nansum(s2) / np.nansum(s1) - 1)
    
    delta_cell = 100 * np.nanmean((s2 - s1) / (s1 + s2))

    # or use a threshold
    threshold = 0.5
    percentage_change_th = 100 * (np.nansum(s2 > threshold) / np.nansum(s1 > threshold) - 1)
    print(species_names[species_indx], percentage_change, percentage_change_th, delta_cell)
    summary_res.append([species_names[species_indx], percentage_change, percentage_change_th, delta_cell])
    
summary_res = pd.DataFrame(summary_res)
summary_res.columns = ["Species", "delta_tot_density", "delta_area_threshold", "mean_rel_change"]
summary_res.to_csv(os.path.join(data_wd, "predicted_sdm_change.txt"), sep="\t", index=False)