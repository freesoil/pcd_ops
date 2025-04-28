
import numpy as np
import copy
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import logging


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
    """
    The function corrects the orientation of normals in a point cloud by comparing them with a target
    point cloud and flipping the normals if necessary.
    
    :param source: The `source` parameter in the `correct_normal_orientation` function is a point cloud
    data structure representing the source point cloud with normals. It contains the points and
    corresponding normals of the source point cloud
    :param target: The `target` parameter in the `correct_normal_orientation` function is expected to be
    a point cloud data structure representing the target geometry with normals. It should contain the
    points and corresponding normals of the target surface
    :param k: The `k` parameter in the `correct_normal_orientation` function represents the number of
    nearest neighbors to consider when finding the closest points in the target point cloud for each
    point in the source point cloud. This parameter is used in the k-nearest neighbors (KNN) algorithm
    to determine the correspondence between, defaults to 2 (optional)
    :return: The function `correct_normal_orientation` returns a point cloud with corrected normal
    orientations. If the error calculated using the original normals is less than the error calculated
    using the negated normals, the function returns a point cloud with the normals negated. Otherwise,
    it returns the original point cloud with its normals unchanged.
    """

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
    """
    The function `compute_normal_error` calculates the average error between corresponding normals of
    source and target point clouds based on nearest neighbors.
    
    :param source: The `compute_normal_error` function calculates the error between the normals of two
    point clouds, `source` and `target`, using the k-nearest neighbors algorithm. The function takes in
    the following parameters:
    :param target: The `compute_normal_error` function you provided calculates the error between the
    normals of source and target point clouds. The function takes in the source and target point clouds
    along with an optional parameter `k` which specifies the number of nearest neighbors to consider
    :param k: The parameter `k` in the `compute_normal_error` function represents the number of nearest
    neighbors to consider when computing the normal error between the source and target point clouds.
    Increasing the value of `k` will result in considering more neighbors for comparison, potentially
    leading to a more accurate estimation of the normal, defaults to 2 (optional)
    :return: The function `compute_normal_error` returns the mean error between the matched normals of
    the source and target point clouds based on the specified nearest neighbors (k) for the target
    points.
    """

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


def registration_pairwise_coarse_multiple_initialization(source, target, max_correspondence_distance_coarse):
    """
    The function `registration_pairwise_coarse_multiple_initialization` performs pairwise coarse
    registration between source and target point clouds using multiple initializations and returns the
    best registration result along with its fitness score.
    
    :param source: The `source` parameter in the `registration_pairwise_coarse_multiple_initialization`
    function represents the point cloud data of the source geometry that you want to align with the
    target geometry. It is typically a NumPy array or an Open3D PointCloud object containing the 3D
    points of the
    :param target: The `target` parameter in the `registration_pairwise_coarse_multiple_initialization`
    function refers to the point cloud data of the target object that you want to align or register the
    `source` point cloud data with. It is used in the registration process to find the transformation
    that aligns the `
    :param max_correspondence_distance_coarse: The parameter `max_correspondence_distance_coarse`
    represents the maximum correspondence distance for coarse registration. It is used in the
    registration process to determine the maximum allowed distance between corresponding points in the
    source and target point clouds during the initial alignment. This parameter helps in filtering out
    outlier correspondences and improving the
    :return: The function `registration_pairwise_coarse_multiple_initialization` returns the best
    registration result (`best_result`) and its corresponding fitness score (`best_fitness`).
    """
    
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

        result = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance=max_correspondence_distance_coarse,
            init=init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        normal_error = compute_normal_error(copy.deepcopy(source).transform(result.transformation), target)
        normal_score = (2.0-normal_error)/2.0

        score = normal_score*0.8 + result.fitness*0.2

        if score > best_fitness:
            best_fitness = score
            best_result = result
    
    return best_result, best_fitness

def registration_pairwise_coarse_to_fine_icp(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine, fitness_threshold=0.7, use_icp_color=True, use_simple_coarse=False):
    """
    The function `registration_pairwise_coarse_to_fine_icp` performs pairwise coarse-to-fine ICP
    registration between a source and target point cloud, with options for using color information and
    multiple initializations.
    
    :param source: source point cloud
    :param target: target point cloud
    :param max_correspondence_distance_coarse: The parameter `max_correspondence_distance_coarse`
    represents the maximum correspondence distance allowed between points in the source and target point
    clouds during the coarse registration step. This distance is used to filter out point
    correspondences that are too far apart to be considered for alignment.
    :param max_correspondence_distance_fine: The parameter `max_correspondence_distance_fine` in the
    `registration_pairwise_coarse_to_fine_icp` function represents the maximum correspondence distance
    allowed between points in the source and target point clouds during the fine registration step.
    :param fitness_threshold: The `fitness_threshold` parameter in the
    `registration_pairwise_coarse_to_fine_icp` function is used to determine the threshold value for the
    fitness score of the final ICP registration. If the fitness is below this threshold, the registration
    is not acceptable.
    :param use_icp_color: The `use_icp_color` parameter in the
    `registration_pairwise_coarse_to_fine_icp` function determines whether to use colored ICP registration.
    :param use_simple_coarse: The `use_simple_coarse` parameter in the
    `registration_pairwise_coarse_to_fine_icp` function determines whether a simple coarse registration
    method should be used or not. If `use_simple_coarse` is set to `True`, the function will perform a
    basic registration using the point to point ICP
    :return: The function `registration_pairwise_coarse_to_fine_icp` returns four values:
    `transformation_icp`, `information_icp`, `icp_fine`, and `flip_normals`.
    """

    source_flipped_normals = copy.deepcopy(source)
    normals = np.asarray(source_flipped_normals.normals)
    normals = -normals
    source_flipped_normals.normals = o3d.utility.Vector3dVector(normals)
    flip_normals = False

    icp_coarse = None
    
    if use_simple_coarse:
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        # print("coarse", icp_coarse.fitness)
    else:
        icp_coarse_1, score_coarse_1 = registration_pairwise_coarse_multiple_initialization(source, target, max_correspondence_distance_coarse)
        icp_coarse_2, score_coarse_2 = registration_pairwise_coarse_multiple_initialization(source_flipped_normals, target, max_correspondence_distance_coarse)
        
        if score_coarse_1 > score_coarse_2:
            icp_coarse = icp_coarse_1
            flip_normals = False
        else:
            icp_coarse = icp_coarse_2
            flip_normals = True
    
        # print("coarse", icp_coarse.fitness, max(score_coarse_1,score_coarse_2))

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
    
    if use_icp_color:

        icp_fine_color = None
        try:
            icp_fine_color = o3d.pipelines.registration.registration_colored_icp(
                    source, target, max_correspondence_distance_fine, icp_fine.transformation,
                    estimation_method = o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                    relative_rmse=1e-6,
                                                                    max_iteration=20))
        except:
            logging.warning("Unable to apply color icp")
        if (icp_fine_color is None):
            return None, None, None, False
        else:
            icp_fine = icp_fine_color
        
        
        # print("color", icp_fine.fitness,icp_fine.inlier_rmse)
        
    if icp_fine.fitness < fitness_threshold:
        return None, None, None, False
    
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    

    return transformation_icp, information_icp, icp_fine, flip_normals


def registration_multiple_coarse_to_fine_icp(pcd_list, voxel_size, fitness_threshold=0.7, use_icp_color=True, use_simple_coarse=False):
    """
    The function `registration_multiple_coarse_to_fine_icp` registers multiple point clouds using
    coarse-to-fine ICP registration with optional color and coarse settings.
    
    :param pcd_list: The `pcd_list` parameter is a list of point cloud data objects that you want to
    register together. Each element in the list represents a point cloud
    :param voxel_size: The `voxel_size` parameter in the `registration_multiple_coarse_to_fine_icp`
    function represents the size of the voxels used in downsampling the point clouds during the
    registration process. It is a crucial parameter that determines the level of detail in the resulting
    registered point cloud. A smaller
    :param fitness_threshold: The `fitness_threshold` parameter in the
    `registration_multiple_coarse_to_fine_icp` function is used as a threshold value to determine the
    quality of the registration alignment between two point clouds during the ICP (Iterative Closest
    Point) registration process. It represents the maximum allowable fitness score for
    :param use_icp_color: The `use_icp_color` parameter in the
    `registration_multiple_coarse_to_fine_icp` function determines whether color information should be
    used during the Iterative Closest Point (ICP) registration process. When `use_icp_color` is set to
    `True`, the color information of, defaults to True (optional)
    :param use_simple_coarse: The `use_simple_coarse` parameter in the
    `registration_multiple_coarse_to_fine_icp` function is a boolean flag that determines whether a
    simple coarse registration method should be used. When `use_simple_coarse` is set to `True`, a
    simpler coarse registration method will be used in, defaults to False (optional)
    :return: The function `registration_multiple_coarse_to_fine_icp` returns two main objects:
    1. `global_pcd`: The globally registered point cloud after registering all the input point clouds in
    the `pcd_list`.
    2. `registered_pcds`: A list containing the registered point clouds after aligning each point cloud
    in the `pcd_list` to the global point cloud.
    3. `transformations`: A list containing the registration transformations.
    """
    registered_pcds = [copy.deepcopy(pcd_list[0])]
    global_pcd = copy.deepcopy(pcd_list[0])
    transformations = [np.identity(4)]
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
            transformation = None
        else:
            new_pcd = copy.deepcopy(pcd_list[i]).transform(transformation_icp)
            transformation = transformation_icp
            if flip_normals:
                normals = np.asarray(new_pcd.normals)
                normals = -normals
                new_pcd.normals = o3d.utility.Vector3dVector(normals)

            new_pcd = correct_normal_orientation(new_pcd, global_pcd, k=5)
            global_pcd += new_pcd

            global_pcd = global_pcd.voxel_down_sample(voxel_size*0.1)

        registered_pcds.append(new_pcd)
        transformations.append(transformation)
    logging.info("Number of point clouds included in the registration process: {}/{}".format(sum(1 for item in registered_pcds if item is not None),len(pcd_list)))
    return global_pcd, registered_pcds, transformations




def compute_overlap_confidence(pcd_list, radius, overlap_threshold=2, use_per_point_cloud_checking=True):
    """
    The function `compute_overlap_confidence` calculates overlap confidence between point clouds based
    on a specified radius and threshold, with an option for per-point cloud checking.
    
    :param pcd_list: The `pcd_list` parameter in the `compute_overlap_confidence` function is a list of
    point cloud data. Each element in the list represents a point cloud, and the function processes
    these point clouds to compute the overlap confidence based on the specified parameters
    :param radius: The `radius` parameter in the `compute_overlap_confidence` function is used to
    specify the radius for the NearestNeighbors algorithm. This algorithm is used to find points within
    a certain radius of a query point. It helps in determining the neighborhood of each point in the
    point clouds
    :param overlap_threshold: The `overlap_threshold` parameter in the `compute_overlap_confidence`
    function is used to specify the minimum number of unique cloud IDs required in the neighborhood of a
    query point to consider it as a confident observation. If the number of unique cloud IDs in the
    neighborhood of a query point is greater than or, defaults to 2 (optional)
    :param use_per_point_cloud_checking: The `use_per_point_cloud_checking` parameter in the
    `compute_overlap_confidence` function is a boolean flag that determines whether or not to perform
    per-point cloud checking. When set to `True`, the function will evaluate the overlap confidence for
    each individual point cloud in `pcd_list`. This, defaults to True (optional)
    :return: The function `compute_overlap_confidence` returns three values:
    1. `obs_confidence_mask`: A boolean mask indicating the points that meet the overlap confidence
    threshold.
    2. `merged_pcd`: The merged and downsampled point cloud used as reference query points.
    3. `labels`: The labels assigned to each point based on the cloud index.
    """

    # Combine all points and label them by cloud index
    all_points = []
    labels = []
    for i, pcd in enumerate(pcd_list):
        pts = np.asarray(pcd.points)
        all_points.append(pts)
        labels.append(np.full(len(pts), i))

    all_points = np.vstack(all_points)
    labels = np.concatenate(labels)

    # Fit NearestNeighbors on all points
    nn = NearestNeighbors(radius=radius, algorithm='kd_tree')
    nn.fit(all_points)

    # Merge and downsample the point clouds to get reference query points
    merged_pcd = pcd_list[0]
    for pcd in pcd_list[1:]:
        merged_pcd += pcd
    query_points = np.asarray(merged_pcd.points)

    # Radius search for each query point
    distances, indices = nn.radius_neighbors(query_points)

    # Count unique cloud IDs in neighborhood
    observation_counts = np.array([
        len(set(labels[inds])) if len(inds) > 0 else 0 for inds in tqdm(indices , desc="Outlier removal based on overlap confidence")
    ])

    # confidence mask
    obs_confidence_mask = observation_counts >= overlap_threshold

    # per point cloud checking
    if use_per_point_cloud_checking:
        for i, pcd in enumerate(pcd_list):
            pcd_mask = labels==i
            ratio = np.count_nonzero(pcd_mask*obs_confidence_mask)/np.count_nonzero(pcd_mask)
            if ratio>0.6:
                obs_confidence_mask[pcd_mask] = True
            if ratio<0.3:
                obs_confidence_mask[pcd_mask] = False


    return obs_confidence_mask, merged_pcd, labels


def compute_visibility_confidence(pcd):
    """
    The function `compute_visibility_confidence` calculates the visibility confidence of points in a
    point cloud based on their visibility from external viewpoints and facing direction.
    
    :param pcd: The `pcd` parameter in the `compute_visibility_confidence` function seems to be a point
    cloud data structure. The function calculates the visibility confidence of points in the point cloud
    from different viewpoints. It computes geometric structures, defines external viewpoints, and then
    checks visibility from each viewpoint.
    :return: The function `compute_visibility_confidence` returns a boolean mask indicating which points
    in the input point cloud `pcd` are visible from at least one viewpoint out of a set of predefined
    external viewpoints.
    """

    # Compute geoemtric structures
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    radius = np.max(distances)*1.5

    # Define external viewpoints
    viewpoints = []
    for theta in np.linspace(0, 2 * np.pi, 16):
        for phi in np.linspace(0, np.pi, 8):
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            viewpoints.append(centroid + np.array([x, y, z]))

    # Initialize visibility counter
    visible_counts = np.zeros(len(pcd.points))

    # Check visibility from each viewpoint and check if the points are facing the view point
    for vp in tqdm(viewpoints, desc="Outlier removal based on visibility confidence"):
        # visibility based on Direct Visibility of Point Sets (Katz et al.)
        _, pt_map = pcd.hidden_point_removal(vp, radius * 70.0) 
        # facing checking
        dot_products = np.einsum('ij,ij->i', points-vp, normals)
        orientation_mask = dot_products <= 0
        vis_mask = np.zeros(len(points),dtype=bool)
        vis_mask[pt_map] = True

        visible_counts[np.where(vis_mask*orientation_mask)[0]] += 1

    # Threshold to keep points visible from at least N views
    min_views = 1
    visible_mask = visible_counts >= min_views

    return visible_mask
