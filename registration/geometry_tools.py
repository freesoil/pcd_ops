
import numpy as np
import copy
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def compute_avg_distance(in_pcd):
    """
    The function `compute_avg_distance` calculates the average distance between points in a point cloud
    after removing repeated elements.
    
    :param in_pcd: Input point cloud
    :return: The function `compute_avg_distance` returns the median of the distances between each point
    and its 5 nearest neighbors in the input point cloud after removing repeated elements through voxel
    downsampling.
    """
    pcd = in_pcd.voxel_down_sample(0.00000001) # remove repeated elements
    points = np.asarray(pcd.points)
    points_knn = NearestNeighbors(n_neighbors=5)
    points_knn.fit(points)
    distances, indices = points_knn.kneighbors(points, return_distance=True)
    distances = distances[:,1:].flatten()
    # print(distances)
    average_distance = np.median(distances)
    return average_distance

def correct_normal_orientation(source, target, k=2):

    out_pcd = copy.deepcopy(source)

    source_points = np.asarray(source.points)
    source_normals = np.asarray(source.normals)
    target_points = np.asarray(target.points)
    target_normals = np.asarray(target.normals)
    target_points_knn = NearestNeighbors(n_neighbors=k)
    target_points_knn.fit(target_points)
    distances, indices = target_points_knn.kneighbors(source_points, return_distance=True)

    sorted_indices_source = np.argsort(distances[:, :].reshape(-1))
    top_n_indices_source = sorted_indices_source[:int(len(sorted_indices_source)*0.5)]

    all_indices = indices[:, :].reshape(-1)
    top_n_indices_target = all_indices[top_n_indices_source]

    source_normals_comp = np.tile(source_normals, (k,1))
    source_normals_comp = source_normals_comp[top_n_indices_source]

    matched_normals_comp = target_normals[top_n_indices_target]

    error_1 = np.mean(np.linalg.norm(matched_normals_comp - source_normals_comp, axis=1))
    error_2 = np.mean(np.linalg.norm(matched_normals_comp - (-source_normals_comp), axis=1))

    if error_2 < error_1:
        out_pcd.normals = o3d.utility.Vector3dVector(-source_normals)

    return out_pcd

def compute_normal_error(source, target, k=2):

    source_points = np.asarray(source.points)
    source_normals = np.asarray(source.normals)
    target_points = np.asarray(target.points)
    target_normals = np.asarray(target.normals)
    target_points_knn = NearestNeighbors(n_neighbors=k)
    target_points_knn.fit(target_points)
    distances, indices = target_points_knn.kneighbors(source_points, return_distance=True)

    sorted_indices_source = np.argsort(distances[:, :].reshape(-1))
    top_n_indices_source = sorted_indices_source[:int(len(sorted_indices_source)*0.5)]

    all_indices = indices[:, :].reshape(-1)
    top_n_indices_target = all_indices[top_n_indices_source]

    source_normals_comp = np.tile(source_normals, (k,1))
    source_normals_comp = source_normals_comp[top_n_indices_source]

    matched_normals_comp = target_normals[top_n_indices_target]

    errors = np.linalg.norm(matched_normals_comp - source_normals_comp, axis=1)

    error = np.mean(errors)

    return error


def registration_preprocessing(pcd, voxel_size):
    """
    The function `registration_preprocessing` downsamples a point cloud, estimates normals, computes
    FPFH features, and returns the processed point cloud and FPFH features.
    
    :param pcd: The `pcd` parameter in the `registration_preprocessing` function is a point
    cloud data structure that contains the 3D points of an object or scene. This data structure is
    commonly used in point cloud processing and registration tasks
    :param voxel_size: The `voxel_size` parameter in the `registration_preprocessing` function is used
    to specify the size of the voxels for downsampling the point cloud. It is a crucial parameter that
    determines the level of detail in the processed point cloud data
    :return: The function `registration_preprocessing` returns two variables: `pcd_down` and `pcd_fpfh`.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def registration_pairwise_feature_and_icp(pcd_source, pcd_target, voxel_size):
    pcd_out = copy.deepcopy(pcd_source)

    pcd_source_down, pcd_source_fpfh = registration_preprocessing(pcd_source, voxel_size)
    pcd_target_down, pcd_target_fpfh = registration_preprocessing(pcd_target, voxel_size)

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(pcd_source_down, pcd_target_down, 
                                                                                             pcd_source_fpfh, pcd_target_fpfh, 
                                                                                             True, voxel_size*5, 
                                                                                             o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
                                                                                             [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size*10)], 
                                                                                             o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999))
    
    result_icp = o3d.pipelines.registration.registration_icp(pcd_source, pcd_target, voxel_size*5, result_ransac.transformation, 
                                                             o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    pcd_out.transform(result_icp.transformation)

    return pcd_out, result_icp.transformation

def registration_multiple_feature_and_icp(pcd_list, voxel_size):
    registered_pcds = [copy.deepcopy(pcd_list[0])]
    global_pcd = copy.deepcopy(pcd_list[0])
    for i in tqdm(range(1,len(pcd_list)), desc="Registering point clouds"):
        new_pcd, _ = registration_pairwise_feature_and_icp(pcd_list[i], global_pcd, voxel_size)
        global_pcd += new_pcd
        registered_pcds.append(new_pcd)
    return global_pcd, registered_pcds


def registration_pairwise_coarse_multiple_initialization(source, target, max_correspondence_distance_coarse):
    
    d = max_correspondence_distance_coarse*0.5

    translations = [np.array([0, 0, 0]), np.array([d, 0, 0]), np.array([-d, 0, 0]), np.array([0, d, 0]),np.array([0, -d, 0]), 
                    np.array([0, 0, d]), np.array([0, 0, -d])]

    initial_transforms = []

    for tx in translations:
        T = np.eye(4)
        T[:3, 3] = tx
        initial_transforms.append(T)

    best_fitness = -1
    best_result = None

    for i, init in enumerate(initial_transforms):
        
        # result = o3d.pipelines.registration.registration_icp(
        #     source, target, max_correspondence_distance=max_correspondence_distance_coarse,
        #     init=init,
        #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
        # )

        result = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance=max_correspondence_distance_coarse,
            init=init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        normal_error = compute_normal_error(copy.deepcopy(source).transform(result.transformation), target)
        normal_score = (2.0-normal_error)/2.0

        score = normal_score*0.8 + result.fitness*0.2

        # print(normal_score, result.fitness)
        # print(init)
        # print(result.transformation)

        if score > best_fitness:
            best_fitness = score
            best_result = result
    
    # source_initializations = [copy.deepcopy(source).transform(tr).paint_uniform_color(np.array([np.random.uniform(),np.random.uniform(),np.random.uniform()])) for tr in initial_transforms]
    # o3d.visualization.draw_geometries([target,copy.deepcopy(source).transform(best_result.transformation)] + source_initializations)
    # o3d.visualization.draw_geometries([target,copy.deepcopy(source).transform(best_result.transformation)])

    return best_result, best_fitness

def registration_pairwise_coarse_to_fine_icp(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine, 
                                             fitness_threshold=0.7, use_icp_color=True, use_simple_coarse=False):

    source_flipped_normals = copy.deepcopy(source)
    normals = np.asarray(source_flipped_normals.normals)
    normals = -normals
    source_flipped_normals.normals = o3d.utility.Vector3dVector(normals)
    flip_normals = False
    # o3d.visualization.draw_geometries([target,source])

    icp_coarse = None
    
    if use_simple_coarse:
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print("coarse", icp_coarse.fitness)
    else:
        icp_coarse_1, score_coarse_1 = registration_pairwise_coarse_multiple_initialization(source, target, max_correspondence_distance_coarse)
        icp_coarse_2, score_coarse_2 = registration_pairwise_coarse_multiple_initialization(source_flipped_normals, target, max_correspondence_distance_coarse)
        
        if score_coarse_1 > score_coarse_2:
            icp_coarse = icp_coarse_1
            flip_normals = False
        else:
            icp_coarse = icp_coarse_2
            flip_normals = True
    
        print("coarse", icp_coarse.fitness, max(score_coarse_1,score_coarse_2))

    icp_fine = None

    icp_fine_1 = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    normal_error_1 = compute_normal_error(copy.deepcopy(source).transform(icp_fine_1.transformation), target)
    normal_score_1 = (2.0-normal_error_1)/2.0
    score_1 = icp_fine_1.fitness*0.2 + normal_score_1*0.8
    
    icp_fine_2 = o3d.pipelines.registration.registration_icp(
        source_flipped_normals, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    normal_error_2 = compute_normal_error(copy.deepcopy(source).transform(icp_fine_2.transformation), target)
    normal_score_2 = (2.0-normal_error_2)/2.0
    score_2 = icp_fine_2.fitness*0.2 + normal_score_2*0.8
    
    if score_1 > score_2:
        icp_fine = icp_fine_1
        flip_normals = False
    else:
        icp_fine = icp_fine_2
        flip_normals = True
    
    print("geometric", icp_fine.fitness,max(score_1,score_2))

    # o3d.visualization.draw_geometries([target,copy.deepcopy(source).transform(icp_fine.transformation)])

    geom_fine_transform = icp_fine.transformation
    
    # o3d.io.write_point_cloud("source_icp_fine.ply",  copy.deepcopy(source).transform(icp_fine.transformation))


    if use_icp_color:
        # icp_color_res_1 = None
        # icp_color_res_2 = None
        # try:
        #     icp_fine_color = o3d.pipelines.registration.registration_colored_icp(
        #             source, target, max_correspondence_distance_fine, icp_fine.transformation,
        #             estimation_method = o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        #             criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
        #                                                             relative_rmse=1e-6,
        #                                                             max_iteration=20))
        #     icp_color_res_1 = icp_fine_color
        # except:
        #     print("Unable to apply color icp (1)")
        
        # try:
        #     icp_fine_color = o3d.pipelines.registration.registration_colored_icp(
        #             source_flipped_normals, target, max_correspondence_distance_fine, icp_fine.transformation,
        #             estimation_method = o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        #             criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
        #                                                             relative_rmse=1e-6,
        #                                                             max_iteration=20))
        #     icp_color_res_2 = icp_fine_color
        # except:
        #     print("Unable to apply color icp (2)")

        # if (icp_color_res_1 is None) and (icp_color_res_2 is None):
        #     return None, None, None, False
        # elif (not (icp_color_res_1 is None)) and (not (icp_color_res_2 is None)):
        #     if icp_color_res_1.fitness > icp_color_res_2.fitness:
        #         icp_fine = icp_color_res_1
        #         flip_normals = False
        #     else:
        #         icp_fine = icp_color_res_2
        #         flip_normals = True
        # elif icp_color_res_1 is None:
        #     icp_fine = icp_color_res_2
        #     flip_normals = True
        # else:
        #     icp_fine = icp_color_res_1
        #     flip_normals = False

        icp_fine_color = None
        try:
            icp_fine_color = o3d.pipelines.registration.registration_colored_icp(
                    source, target, max_correspondence_distance_fine, icp_fine.transformation,
                    estimation_method = o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                    relative_rmse=1e-6,
                                                                    max_iteration=20))
        except:
            print("Unable to apply color icp")
        if (icp_fine_color is None):
            return None, None, None, False
        else:
            icp_fine = icp_fine_color
        
        
        print("color", icp_fine.fitness,icp_fine.inlier_rmse)
        
        # o3d.io.write_point_cloud("source_icp_fine_color.ply",  copy.deepcopy(source).transform(icp_fine.transformation))
    # if icp_fine.fitness < 0.8:
    #     o3d.io.write_point_cloud("target.ply",  target)
    #     o3d.io.write_point_cloud("source.ply",  source)
    #     o3d.io.write_point_cloud("source_icp_coarse.ply",  copy.deepcopy(source).transform(icp_coarse.transformation))
    #     o3d.io.write_point_cloud("source_icp_fine_geom.ply",  copy.deepcopy(source).transform(geom_fine_transform))
    #     o3d.io.write_point_cloud("source_icp_fine_color.ply",  copy.deepcopy(source).transform(icp_fine.transformation))


    if icp_fine.fitness < fitness_threshold:
        return None, None, None, False
    
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    

    return transformation_icp, information_icp, icp_fine, flip_normals


def registration_multiple_coarse_to_fine_icp(pcd_list, voxel_size, fitness_threshold=0.7, use_icp_color=True, use_simple_coarse=False):
    registered_pcds = [copy.deepcopy(pcd_list[0])]
    global_pcd = copy.deepcopy(pcd_list[0])
    for i in tqdm(range(1,len(pcd_list)), desc="Registering point clouds"):
        if len(pcd_list[i].points)>len(global_pcd.points):
            transformation_icp, information_icp, icp, flip_normals = registration_pairwise_coarse_to_fine_icp(global_pcd, pcd_list[i], voxel_size*15.0, 
                                                                                                              voxel_size*2.0, fitness_threshold, 
                                                                                                              use_icp_color, use_simple_coarse)
            if transformation_icp is None:
                transformation_icp, information_icp, icp, flip_normals = registration_pairwise_coarse_to_fine_icp(pcd_list[i], global_pcd, voxel_size*15.0, 
                                                                                                                  voxel_size*2.0, fitness_threshold, 
                                                                                                                  use_icp_color, use_simple_coarse)
            else:
                transformation_icp = np.linalg.inv(transformation_icp)
        else:
            transformation_icp, information_icp, icp, flip_normals = registration_pairwise_coarse_to_fine_icp(pcd_list[i], global_pcd, voxel_size*15.0, 
                                                                                                              voxel_size*2.0, fitness_threshold, 
                                                                                                              use_icp_color, use_simple_coarse)
        
        if transformation_icp is None:
            new_pcd = None
        else:
            new_pcd = copy.deepcopy(pcd_list[i]).transform(transformation_icp)
            if flip_normals:
                normals = np.asarray(new_pcd.normals)
                normals = -normals
                new_pcd.normals = o3d.utility.Vector3dVector(normals)

            new_pcd = correct_normal_orientation(new_pcd, global_pcd, k=5)
            global_pcd += new_pcd

            # partial_pcd = o3d.geometry.PointCloud()
            # partial_pcd_list = registered_pcds[max(0,i-10):i] + [new_pcd]
            # partial_pcd_list = [x for x in partial_pcd_list if x is not None]
            # for j in range(len(partial_pcd_list)):
            #     partial_pcd += partial_pcd_list[j]

            global_pcd = global_pcd.voxel_down_sample(voxel_size*0.1)
            # partial_pcd.orient_normals_consistent_tangent_plane(k=50)

        registered_pcds.append(new_pcd)
    print("Number of point clouds considered: {}/{}".format(sum(1 for item in registered_pcds if item is not None),len(pcd_list)))
    return global_pcd, registered_pcds


def pose_graph_estimation(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine, use_icp_color=False):
    pose_graph = o3d.pipelines.registration.PoseGraph()

    n_pcds = len(pcds)

    for i in range(n_pcds):
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))

    for it_source_id in range(n_pcds):
        for it_target_id in range(it_source_id + 1, min(n_pcds,it_source_id + 10)):
        # for it_target_id in range(it_source_id + 1, n_pcds):
            source_id = it_source_id
            target_id = it_target_id
            if len(pcds[target_id].points) < len(pcds[source_id].points):
                source_id = it_target_id
                target_id = it_source_id

            transformation_icp, information_icp, icp = registration_pairwise_coarse_to_fine_icp(
                    pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine, use_icp_color)
            
            if not (transformation_icp is None):
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=False))
            else:
                print("Unable to perform registration between {} and {}".format(source_id,target_id))


            # transformation_icp, information_icp, icp = registration_pairwise_coarse_to_fine_icp(
            #         pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine, use_icp_color)
            
            # transformation_icp_opp, information_icp_opp, icp_opp = registration_pairwise_coarse_to_fine_icp(
            #         pcds[target_id], pcds[source_id], max_correspondence_distance_coarse, max_correspondence_distance_fine, use_icp_color)
            
            # if not (transformation_icp is None):
            #     flag = transformation_icp_opp is None
            #     if not flag:
            #         flag = icp.fitness > icp_opp.fitness
            #     if flag:
            #         pose_graph.edges.append(
            #             o3d.pipelines.registration.PoseGraphEdge(source_id,
            #                                             target_id,
            #                                             transformation_icp,
            #                                             information_icp,
            #                                             uncertain=True))
            # else:
            #     print("Unable to perform registration between {} and {}".format(source_id,target_id))

            

            # if not (transformation_icp_opp is None):
            #     flag = transformation_icp is None
            #     if not flag:
            #         flag = icp_opp.fitness >= icp.fitness
            #     if flag:
            #         pose_graph.edges.append(
            #             o3d.pipelines.registration.PoseGraphEdge(target_id,
            #                                             source_id,
            #                                             transformation_icp_opp,
            #                                             information_icp_opp,
            #                                             uncertain=True))
            # else:
            #     print("Unable to perform registration between {} and {}".format(source_id,target_id))

            # if target_id == source_id + 1:  # odometry case    
            #     pose_graph.edges.append(
            #         o3d.pipelines.registration.PoseGraphEdge(source_id,
            #                                        target_id,
            #                                        transformation_icp,
            #                                        information_icp,
            #                                        uncertain=False))
            # else:  # loop closure case
            #     pose_graph.edges.append(
            #         o3d.pipelines.registration.PoseGraphEdge(source_id,
            #                                        target_id,
            #                                        transformation_icp,
            #                                        information_icp,
            #                                        uncertain=True))
    return pose_graph

def registration_multiple_pose_graph(pcds, voxel_size, use_icp_color=False):
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pcds_down = [pcd.voxel_down_sample(voxel_size) for pcd in pcds]
    # o3d.visualization.draw_geometries(pcds_down)

    print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    pose_graph = pose_graph_estimation(pcds_down,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine,
                                   use_icp_color)

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)

    print("Make a combined point cloud")

    pcd_combined = o3d.geometry.PointCloud()
    registered_pcds = []
    ignored_pcds = []
    for point_id in range(len(pcds)):
        t_pcd = copy.deepcopy(pcds[point_id])
        try:
            transform = pose_graph.nodes[point_id].pose
        except:
            print("Ignoring point cloud {}".format(point_id))
            continue
        t_pcd.transform(pose_graph.nodes[point_id].pose)
        pcd_combined += t_pcd
        registered_pcds.append(t_pcd)
    
    return pcd_combined, registered_pcds

def registration_pairwise_color(source, target):
    pcd_out = copy.deepcopy(source)
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)

    pcd_out.transform(result_icp.transformation)
    transformation_icp_color = result_icp.transformation
    
    return pcd_out, transformation_icp_color

def registration_multiple_color(pcd_list):
    registered_pcds = [copy.deepcopy(pcd_list[0])]
    global_pcd = copy.deepcopy(pcd_list[0])
    for i in tqdm(range(1,len(pcd_list)), desc="Registering point clouds"):
        new_pcd, _ = registration_pairwise_color(pcd_list[i], global_pcd)
        global_pcd += new_pcd
        registered_pcds.append(new_pcd)
    return global_pcd, registered_pcds

def registration_multiple_icp_fine(pcd_list, voxel_size):
    registered_pcds = [copy.deepcopy(pcd_list[0])]
    global_pcd = copy.deepcopy(pcd_list[0])
    for i in tqdm(range(1,len(pcd_list)), desc="Registering point clouds"):
        source = pcd_list[i].voxel_down_sample(voxel_size)
        target = global_pcd.voxel_down_sample(voxel_size)
        icp_fine = o3d.pipelines.registration.registration_icp(source, target, voxel_size*1.5, np.identity(4), o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transformation_icp = icp_fine.transformation
        new_pcd = copy.deepcopy(pcd_list[i])
        new_pcd.transform(transformation_icp)
        global_pcd += new_pcd
        registered_pcds.append(new_pcd)
    return global_pcd, registered_pcds

def compute_confidence(
    merged_pcd: o3d.geometry.PointCloud,
    list_of_pcds: list[o3d.geometry.PointCloud],
    radius: float = 0.02,
    alpha: float = 0.5  # balance weight between normal and distance
) -> np.ndarray:
    """
    Computes confidence score per point in the merged point cloud.

    Parameters:
    - merged_pcd: Aligned point cloud (Nx3)
    - list_of_pcds: List of aligned original scans
    - radius: Search radius for local support
    - alpha: Weight for normal agreement vs. distance confidence [0–1]

    Returns:
    - confidence: N-element numpy array of confidence scores
    """
    merged_points = np.asarray(merged_pcd.points)
    merged_normals = np.asarray(merged_pcd.normals)
    confidence = np.zeros(len(merged_points))

    for scan in list_of_pcds:
        tree = o3d.geometry.KDTreeFlann(scan)
        scan_points = np.asarray(scan.points)
        scan_normals = np.asarray(scan.normals)

        for i, (p, n) in enumerate(zip(merged_points, merged_normals)):
            [_, idxs, dists] = tree.search_radius_vector_3d(p, radius)
            if len(idxs) == 0:
                continue

            scan_neighbors = scan_points[idxs]
            scan_normals_neighbors = scan_normals[idxs]

            # Distance confidence (inverse of distance)
            dists = np.sqrt(np.asarray(dists))
            dist_conf = 1.0 / (dists + 1e-6)  # avoid div by 0
            dist_conf /= np.max(dist_conf)  # normalize

            # Normal agreement (dot product with merged normal)
            norm_dot = np.dot(scan_normals_neighbors, n)
            norm_conf = np.abs(norm_dot)  # [0, 1]

            # Weighted score
            local_conf = alpha * norm_conf + (1 - alpha) * dist_conf
            local_conf = dist_conf
            confidence[i] += np.mean(local_conf)

    # Normalize final confidence
    if len(list_of_pcds) > 0:
        confidence /= len(list_of_pcds)
    return confidence