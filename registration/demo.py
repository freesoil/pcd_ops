import open3d as o3d
import os
import matplotlib.pyplot as plt
import numpy as np

import geometry_tools as geomtools
import visualization_tools as vistools

def safe_make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


# def registration_pipeline(pcd_list, voxel_size):
#     all_merged_pcds = []
#     for i in range(0,len(pcd_list),5):
#         local_pcd_list = pcd_list[i:min(len(pcd_list),i+6)]
#         merged_pcd, _ = geomtools.registration_multiple_pose_graph(local_pcd_list, voxel_size=voxel_size, use_icp_color=True)
#         all_merged_pcds.append(merged_pcd)
#         o3d.io.write_point_cloud("part{}.ply".format(i), merged_pcd)
#     full_merged_pcd, _ = geomtools.registration_multiple_pose_graph(all_merged_pcds, voxel_size=voxel_size, use_icp_color=False)
#     return full_merged_pcd


if __name__ == '__main__':

    print("Point cloud alignment")

    # Set the folder name within data/input_data. This name will be used to export the results in data/results_data/<name>
    
    #test_name = "hair_dryer"
    #test_name = "hair_dryer_part_1"
    test_name = "20250403_yard_10fps_vslam_full"
    test_name = "gondola2"
    test_name = "gondola_move_marker"

    # Set registration options
    fitness_threshold = 0.7 # 0.7 for objects ith high overlap like the dryer. 0.4 for large scenes with low overlap between sections.
    use_simple_coarse = False # False for objects. True for scenes that present several regions of floor.


    input_folder_path = "data/input_data/{}".format(test_name)
    output_folder_path = "data/results/{}".format(test_name)

    safe_make_folder(output_folder_path)

    filenames = [os.path.join(input_folder_path, f) for f in os.listdir(input_folder_path) if f.endswith((".ply",".pcd"))]
    # filenames = filenames[0:25]

    pcd_list = []

    for filename in filenames:
        pcd = o3d.io.read_point_cloud(filename)
        pcd = pcd.voxel_down_sample(0.00000001) # removing repeated points
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_list.append(pcd)
        if not pcd.has_normals():
            print("No normal information for {}".format(filename))
        if not pcd.has_colors():
            print("No color information for {}".format(filename))

    avg_distance = np.median([geomtools.compute_avg_distance(pcd) for pcd in pcd_list])
    voxel_size_1 = avg_distance*2.0
    voxel_size_2 = avg_distance*1.5
    print("size1: {}, size2 {}".format(voxel_size_1,voxel_size_2))

    new_pcds = []
    for i in range(len(pcd_list)):
        pcd = pcd_list[i]
        labels = np.array(pcd.cluster_dbscan(eps=avg_distance*25.0, min_points=10, print_progress=True))
        largest_label = labels[labels >= 0].max(axis=0, initial=0)
        counts = np.bincount(labels[labels >= 0])
        if len(counts) == 0:
            continue
        largest_label = np.argmax(counts)
        indices = np.where(labels == largest_label)[0]
        pcd = pcd.select_by_index(indices)
        if not pcd.has_normals():
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(k=50)
            pcd.normalize_normals()
        new_pcds.append(pcd)

    pcd_list = new_pcds
        

    

    # full_merged_pcd = registration_pipeline(pcd_list, voxel_size_1)
    # o3d.io.write_point_cloud(os.path.join(output_folder_path,"registration_result.ply"), full_merged_pcd)

    # full_merged_pcd, registered_pcds = geomtools.registration_multiple_pose_graph(pcd_list, voxel_size=voxel_size_1, use_icp_color=True)
    # o3d.io.write_point_cloud(os.path.join(output_folder_path,"registration_result.ply"), full_merged_pcd)
    # for i in range(len(registered_pcds)):
    #     o3d.io.write_point_cloud(os.path.join(output_folder_path,os.path.basename(filenames[i])), registered_pcds[i])

    full_merged_pcd2, registered_pcds2 = geomtools.registration_multiple_coarse_to_fine_icp(pcd_list, voxel_size=voxel_size_1, fitness_threshold=fitness_threshold, use_icp_color=True,
                                                                                            use_simple_coarse=use_simple_coarse)
    o3d.io.write_point_cloud(os.path.join(output_folder_path,"registration_result_full.ply"), full_merged_pcd2)
    for i in range(len(registered_pcds2)):
        if not (registered_pcds2[i] is None):
            o3d.io.write_point_cloud(os.path.join(output_folder_path,os.path.basename(filenames[i])), registered_pcds2[i])

    # # out_pcd, registered_pcds = geomtools.registration_multiple_feature_and_icp(pcd_list, voxel_size=0.05)
    # # merged_pcd, registered_pcds = geomtools.registration_multiple_pose_graph(pcd_list, voxel_size=voxel_size_1*2.0, use_icp_color=False)
    # merged_pcd, registered_pcds = geomtools.registration_multiple_pose_graph(pcd_list, voxel_size=voxel_size_1, use_icp_color=True)
    # # merged_pcd, registered_pcds = geomtools.registration_multiple_pose_graph(registered_pcds, voxel_size=voxel_size_1, use_icp_color=True)
    # o3d.io.write_point_cloud(os.path.join(output_folder_path,"registration_result.ply"), merged_pcd)

    # o3d.io.write_point_cloud(os.path.join(output_folder_path,"registration_result_colorized.ply"), vistools.merge_and_colorize(registered_pcds))

    # merged_pcd_refined, registered_pcds_refined = geomtools.registration_multiple_icp_fine(registered_pcds, voxel_size=voxel_size_2)
    # o3d.io.write_point_cloud(os.path.join(output_folder_path,"registration_refined_result.ply"), merged_pcd_refined)

    # o3d.io.write_point_cloud(os.path.join(output_folder_path,"registration_refined_result_colorized.ply"), vistools.merge_and_colorize(registered_pcds_refined))

    # # confidence = geomtools.compute_confidence(merged_pcd, registered_pcds)
    # # print(confidence.max(),confidence.mean(),confidence.min())

    # # colors = plt.get_cmap("viridis")(confidence / np.max(confidence))[:, :3]
    # # merged_pcd.colors = o3d.utility.Vector3dVector(colors)

    # # o3d.io.write_point_cloud(os.path.join(output_folder_path,"registration_confidence_result.ply"), merged_pcd)
