import numpy as np
import copy
import open3d as o3d


def merge_and_colorize(pcd_list):
    global_pcd = o3d.geometry.PointCloud()
    for i in range(len(pcd_list)):
        pcd = copy.deepcopy(pcd_list[i])
        color = np.random.rand(1, 3)
        colors = np.tile(color, (np.asarray(pcd.points).shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        global_pcd += pcd
    return global_pcd