import open3d as o3d
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import logging
import copy
from tqdm import tqdm

import geometry_tools as geomtools
import visualization_tools as vistools

def safe_make_folder(folder_name):
    """
    The function `safe_make_folder` creates a folder if it does not already exist.
    
    :param folder_name: The `safe_make_folder` function takes a `folder_name` parameter, which is the
    name of the folder that you want to create. If the folder does not already exist, the function will
    create it using `os.makedirs(folder_name)`
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def clear_directory(path):
    """
    The function `clear_directory` deletes all files and folders within a specified directory path.
    
    :param path: The `path` parameter in the `clear_directory` function is the directory path that you
    want to clear of all its contents (files and subdirectories). The function checks if the path exists
    and is a directory. If it is a directory, it iterates over all items in the directory and deletes
    :return: If the `path` provided is not a directory, the function will return without performing any
    further actions.
    """
    if not os.path.isdir(path):
        return
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # remove file or link
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # remove folder
        except Exception as e:
            logging.error(f'Failed to delete {item_path}. Reason: {e}')

# Data used for each execution (for each dataset)
class ProcessingData():
    def __init__(self):
        self.output_folder_path = None
        self.input_filenames = None
        self.input_pcds = None
        self.preprocessed_pcds = None

        self.registration_pcds = None
        self.registration_combined = None
        self.registration_transformations = None
        self.registration_voxel_size = None

        self.outlier_removal_radius = None
        self.outlier_removal_combined = None
        self.outlier_removal_clean_1_overlap = None
        self.outlier_removal_clean_2_visibility = None
        self.outlier_removal_result = None

        self.completion_edge_length = None
        self.completion_chull = None
        self.completion_envelope = None
        self.completion_selected_pcd = None

        self.meshing_voxel_size_new_points = None
        self.meshing_points = None
        self.meshing_result = None

    def load_from_filenames(self, filenames):
        
        selected_filenames = []
        selected_pcds = []

        for filename in tqdm(filenames, desc="Reading input files"):
            try:
                pcd = o3d.io.read_point_cloud(filename)
                assert (len(pcd.points) == len(pcd.colors)) != 0
                selected_filenames.append(filename)
                selected_pcds.append(pcd)
            except:
                logging.warning("Something went wrong when reading {}. The point clouds should have at least one point and color information".format(filename))
        
        self.input_filenames = selected_filenames
        self.input_pcds = selected_pcds

    def load_from_folder(self, folder_path, extensions=(".ply",".pcd")):

        filenames = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extensions)]
        self.load_from_filenames(filenames)

        if self.output_folder_path is None:
            self.output_folder_path = os.path.join("data/results", os.path.basename(os.path.normpath(folder_path)))

# Pipeline for the full processing
class ProcessingPipeline():
    def __init__(self):
        self.param_registration_use_icp_color = True
        self.param_registration_fitness_threshold = 0.7
        self.param_registration_use_simple_coarse = False

        self.param_outlier_removal_use_visibility_confidence = True
        self.param_outlier_removal_use_per_point_cloud_checking = True

        self.param_completion_envelope_iterations = 20

        self.param_export_intermediate_results = True

    def prepare(self):
        pass

    def preprocess_input_data(self, processing_data:ProcessingData):

        preprocessed_pcds = []

        for i in tqdm(range(len(processing_data.input_pcds)), desc="Preprocessing step 1"):
            pcd = processing_data.input_pcds[i]
            pcd = pcd.voxel_down_sample(0.00000001) # removing repeated points
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            preprocessed_pcds.append(pcd)

        avg_distance = np.median([geomtools.compute_avg_distance(pcd) for pcd in preprocessed_pcds])
        processing_data.registration_voxel_size = avg_distance*2.0
        processing_data.outlier_removal_radius = avg_distance*2.5
        processing_data.completion_edge_length = avg_distance*3.0
        processing_data.meshing_voxel_size_new_points = avg_distance*5.0

        for i in tqdm(range(len(preprocessed_pcds)), desc="Preprocessing step 2"):
            pcd = preprocessed_pcds[i]
            labels = np.array(pcd.cluster_dbscan(eps=avg_distance*25.0, min_points=10, print_progress=False))
            largest_label = labels[labels >= 0].max(axis=0, initial=0)
            counts = np.bincount(labels[labels >= 0])
            largest_label = np.argmax(counts)
            indices = np.where(labels == largest_label)[0]
            pcd = pcd.select_by_index(indices)
            if not pcd.has_normals():
                pcd.estimate_normals()
                pcd.orient_normals_consistent_tangent_plane(k=50)
                pcd.normalize_normals()
            preprocessed_pcds[i] = pcd

        processing_data.preprocessed_pcds = preprocessed_pcds
        
    def registration_pipeline(self, processing_data:ProcessingData): # registration pipeline

        full_merged_pcd, registered_pcds, transformations = geomtools.registration_multiple_coarse_to_fine_icp(processing_data.preprocessed_pcds, 
                                                                                                               voxel_size=processing_data.registration_voxel_size, 
                                                                                                               fitness_threshold=self.param_registration_fitness_threshold, 
                                                                                                               use_icp_color=self.param_registration_use_icp_color,
                                                                                                               use_simple_coarse=self.param_registration_use_simple_coarse)
        
        processing_data.registration_combined = full_merged_pcd
        processing_data.registration_pcds = registered_pcds
        processing_data.registration_transformations = transformations


        if self.param_export_intermediate_results:
            results_folder = os.path.join(processing_data.output_folder_path,"registration")
            safe_make_folder(results_folder)
            o3d.io.write_point_cloud(os.path.join(results_folder,"combined.ply"), processing_data.registration_combined)
            for i in range(len(processing_data.registration_pcds)):
                if not (processing_data.registration_pcds[i] is None):
                    o3d.io.write_point_cloud(os.path.join(results_folder,os.path.basename(processing_data.input_filenames[i])), processing_data.registration_pcds[i])

    def outlier_removal_pipeline(self, processing_data:ProcessingData): # outlier removal pipeline

        pcd_list = []
        for i in range(len(processing_data.registration_pcds)):
            if not (processing_data.registration_pcds[i] is None):
                pcd_list.append(processing_data.registration_pcds[i])

        overlap_confidence_mask, merged_pcd, labels = geomtools.compute_overlap_confidence(pcd_list,
                                                                                           processing_data.outlier_removal_radius,
                                                                                           use_per_point_cloud_checking=self.param_outlier_removal_use_per_point_cloud_checking)
        
        clean_1_overlap = merged_pcd.select_by_index(np.where(overlap_confidence_mask)[0])
        
        clean_2_visibility = clean_1_overlap

        if self.param_outlier_removal_use_visibility_confidence:
            visibility_confidence_mask = geomtools.compute_visibility_confidence(clean_1_overlap)
            clean_2_visibility = clean_1_overlap.select_by_index(np.where(visibility_confidence_mask)[0])

        result = copy.deepcopy(clean_2_visibility)
        result.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        result, ind = result.remove_radius_outlier(nb_points=5, radius=processing_data.outlier_removal_radius)

        processing_data.outlier_removal_combined = merged_pcd
        processing_data.outlier_removal_clean_1_overlap = clean_1_overlap
        processing_data.outlier_removal_clean_2_visibility = clean_2_visibility
        processing_data.outlier_removal_result = result

        if self.param_export_intermediate_results:
            results_folder = os.path.join(processing_data.output_folder_path,"outlier_removal")
            safe_make_folder(results_folder)
            o3d.io.write_point_cloud(os.path.join(results_folder,"combined.ply"), processing_data.outlier_removal_combined)
            o3d.io.write_point_cloud(os.path.join(results_folder,"clean_1_overlap.ply"), processing_data.outlier_removal_clean_1_overlap)
            o3d.io.write_point_cloud(os.path.join(results_folder,"clean_2_visibility.ply"), processing_data.outlier_removal_clean_2_visibility)
            o3d.io.write_point_cloud(os.path.join(results_folder,"result.ply"), processing_data.outlier_removal_result)
            

    def completion_pipeline(self, processing_data:ProcessingData):
        selected_pcd, envelope, chull, envelopes = geomtools.envelope_computation(processing_data.outlier_removal_result, processing_data.completion_edge_length)
        processing_data.completion_chull = chull
        processing_data.completion_envelope = envelope
        processing_data.completion_selected_pcd = selected_pcd
        
        if self.param_export_intermediate_results:
            results_folder = os.path.join(processing_data.output_folder_path,"completion")
            safe_make_folder(results_folder)
            o3d.io.write_point_cloud(os.path.join(results_folder,"selected_pcd.ply"), processing_data.completion_selected_pcd)
            o3d.io.write_triangle_mesh(os.path.join(results_folder,"chull.ply"), processing_data.completion_chull)
            o3d.io.write_triangle_mesh(os.path.join(results_folder,"envelope.ply"), processing_data.completion_envelope)
            for i in range(len(envelopes)):
                o3d.io.write_triangle_mesh(os.path.join(results_folder,"envelope_{}.ply".format(i)), envelopes[i])


    def meshing_pipeline(self, processing_data:ProcessingData):
        processing_data.meshing_points = processing_data.outlier_removal_result + processing_data.completion_selected_pcd.voxel_down_sample(processing_data.meshing_voxel_size_new_points)
        processing_data.meshing_result = geomtools.mesh_reconstruction(processing_data.meshing_points)
        if self.param_export_intermediate_results:
            results_folder = os.path.join(processing_data.output_folder_path,"meshing")
            safe_make_folder(results_folder)
            o3d.io.write_point_cloud(os.path.join(results_folder,"points.ply"), processing_data.meshing_points)
            o3d.io.write_triangle_mesh(os.path.join(results_folder,"result.ply"), processing_data.meshing_result)
            


    def full_pipeline(self, processing_data:ProcessingData): # full pipeline execution
        clear_directory(processing_data.output_folder_path)
        self.preprocess_input_data(processing_data)
        self.registration_pipeline(processing_data)
        self.outlier_removal_pipeline(processing_data)
        self.completion_pipeline(processing_data)
        self.meshing_pipeline(processing_data)
