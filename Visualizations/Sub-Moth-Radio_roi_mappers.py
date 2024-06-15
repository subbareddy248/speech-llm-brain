#!/usr/bin/env python
# coding: utf-8

from voxelwise_tutorials.viz import plot_flatmap_from_mapper, _map_to_2d_cmap, plot_2d_flatmap_from_mapper
import numpy as np
import cortex
import os
import matplotlib.pyplot as plt


subject_scores_reading = []
subjects = ['01','02','03','05','07','08']
for eachsub in subjects:
    ceiling_voxcorrs = np.load('../Noise_Ceiling/reading/subject_'+str(eachsub)+'_kernel_ridge.npy')
    subject_scores_reading.append(ceiling_voxcorrs)

subject_scores_listening = []
subjects = ['01','02','03','05','07','08']
for eachsub in subjects:
    ceiling_voxcorrs = np.load('../Noise_Ceiling/listening/subject_'+str(eachsub)+'_kernel_ridge.npy')
    subject_scores_listening.append(ceiling_voxcorrs)


# # Noise Ceiling Voxels 2D Colormaps


plt.rcParams.update({'font.size':20})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
for i in np.arange(6):
    directory = './'
    mapper_file = os.path.join(directory, "mappers", f"subject{subjects[i]}_mappers.hdf")
    ax = plot_2d_flatmap_from_mapper(subject_scores_listening[i],subject_scores_reading[i],
                                 mapper_file, vmin=0, vmax=0.4, vmin2=0,
                                 vmax2=0.4, label_1='Listening',
                                 label_2='Reading',colorbar_location=(.34, .85, .3, .2),cmap='PU_RdBu_covar', with_curvature=True, with_rois=False)
    plt.savefig('subject_noiseceiling_final_2dcolormap_'+subjects[i]+'.jpg', bbox_inches='tight')
    #plt.show()


# # Noise Ceiling Voxels 1D Colormap


subject = '05'
directory = './'
mapper_file = os.path.join(directory, "mappers", f'subject{subject}_mappers.hdf')
plot_flatmap_from_mapper(reading_voxels, mapper_file, vmin=0, vmax=0.7, with_rois=True)
plt.show()


from voxelwise_tutorials.io import load_hdf5_sparse_array
plt.rcParams.update({'font.size':20})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
for i in np.arange(5,6):
    directory = './'
    mapper_file = os.path.join(directory, "mappers", f"subject{subjects[i]}_mappers.hdf")
    voxel_to_fsaverage = load_hdf5_sparse_array(mapper_file, key='voxel_to_fsaverage')
    proj_voxels = voxel_to_fsaverage @ subject_scores_listening[i]
    proj_voxels1 = voxel_to_fsaverage @ subject_scores_reading[i]
    #proj_voxels[np.where(proj_voxels<=0)[0]] = 0
    #proj_voxels1[np.where(proj_voxels1<=0)[0]] = 0
#     ax = plot_2d_flatmap_from_mapper(subject_scores_listening[i],subject_scores_reading[i],
#                                  mapper_file, vmin=0, vmax=0.4, vmin2=0,
#                                  vmax2=0.4, label_1='Listening',
#                                  label_2='Reading',colorbar_location=(.34, .85, .3, .2),cmap='RdBu_covar',  with_curvature=True)
    #plt.savefig('subject_noiseceiling_2dcolormap_'+subjects[i]+'.jpg', bbox_inches='tight')
    #plt.show()
    vertex_data = cortex.Vertex2D(proj_voxels, proj_voxels1, 'fsaverage',vmin=0,vmax=0.4,vmin2=0,vmax2=0.4,cmap="PU_RdBu_covar",colorbar_location=(.34, .85, .3, .2),with_labels=False)
    cortex.quickshow(vertex_data)
    plt.show()
    break

cortex.webgl.show(data=vertex_data)


# this opens a new tab and then saves the lateral and medial views
list_angles = ['left', 'right', 'medial_pivot']
cortex.export.save_3d_views(volume=vertex_data,
                    base_name='./2dmap',
                    list_angles=list_angles,
                    list_surfaces=['inflated'] * len(list_angles),
                    viewer_params=dict(labels_visible=[],
                        overlays_visible=[],
                        linewidth=3))