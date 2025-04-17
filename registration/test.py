import open3d as o3d
import os
import matplotlib.pyplot as plt
import numpy as np
import copy

import geometry_tools as geomtools
import visualization_tools as vistools

def safe_make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

if __name__ == '__main__':

    voxel_size = 0.0022772667648581504
    avg_distance = voxel_size/2.0

    target = o3d.io.read_point_cloud("test_target.ply")
    source = o3d.io.read_point_cloud("test_source.ply")
    source = source.voxel_down_sample(0.00000001) # removing repeated points
    source, ind = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    labels = np.array(source.cluster_dbscan(eps=avg_distance*25.0, min_points=10, print_progress=True))
    largest_label = labels[labels >= 0].max(axis=0, initial=0)
    counts = np.bincount(labels[labels >= 0])
    largest_label = np.argmax(counts)
    indices = np.where(labels == largest_label)[0]
    source = source.select_by_index(indices)
    if not source.has_normals():
        source.estimate_normals()
        source.orient_normals_consistent_tangent_plane(k=50)
        source.normalize_normals()

    max_correspondence_distance_coarse = voxel_size*15.0
    max_correspondence_distance_fine = voxel_size*2.0
    transformation_icp, information_icp, icp_fine, flip_normals = geomtools.registration_pairwise_coarse_to_fine_icp(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine, use_icp_color=True)

    o3d.visualization.draw_geometries([target,copy.deepcopy(source).transform(transformation_icp)])
