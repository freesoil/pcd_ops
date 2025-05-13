# Point Cloud Processing

This project focuses on registering and optimizing a set of point clouds for the generation of clean mesh models. These steps were suggested for the development:

1. Point cloud registration
2. Merged point cloud confidence analysis and outlier removal
3. Point cloud completion using ML
4. Mesh generation 

## Getting started

### Installing Anaconda environment

We can use Anaconda to set an environment.

```bash
conda create -n pcp_env python=3.11
conda activate pcp_env
```

Installing some dependencies using conda (optional for GPU usage).

```bash
conda install -c conda-forge cudatoolkit=11.8 cudnn -y
```

Then, locate the project's root directory and use pip to install the requirements (`requirements.txt`).

```bash
python -m pip install --upgrade pip --user
pip install -r requirements.txt
```

## Running

To run the full processing pipeline that consists of registration ,outlier removal, completion, and meshing subpipelines, you can execute the process as follows:

```bash
python pcp_app.py data/input_data/hair_dryer_part_1
```

This will run the full pipeline using as input the PLY files found in data/input_data/hair_dryer_part_1, and exporting the results to data/results/hair_dryer_part_1, where you will find subfolders containing the results of each subpipeline. For the registration, the combined (registered) point cloud and the isolated registered point clouds will be exported. Notice that just the point clouds that presented an acceptable registration will be considered. For the outlier removal, the results of each step of the algorithm will be exported under the following names: clean_1_overlap.ply, clean_2_visibility.ply, and result.ply. For the completion method, the envelopes computed for each iterations (envelope_X.ply), the final envelope (envelpe.ply), and the selected points (selected_pcd.ply)  will be exported. For the meshing method, the combined point cloud (points.ply) and the meshing result (result.ply) will be exported. These results are grouped in folders with the corresponding subpipelines names, i.e. registration, outlier_removal, completion, and meshing.

The proposed approach relies on certain assumptions about the input data: each point cloud should contain color information and be part of a sequential acquisition, ensuring substantial overlap with preceding scans. This sequence order is represented by the corresponding positions of an input list of point clouds (actually sorted by their filenames). 

To export the results to a custom folder you can use the parameter -o as follows:

```bash
python pcp_app.py data/input_data/hair_dryer_part_1 -o data/results/custom_folder_for_hair_dryer_part_1
```

where the results will be exported to data/results/custom_folder_for_hair_dryer_part_1.

If you want to use custom parameters, you can set them as follows:

```bash
python pcp_app.py data/input_data/20250403_yard_10fps_vslam --param_registration_fitness_threshold 0.4 --param_registration_use_simple_coarse 1 --param_outlier_removal_use_per_point_cloud_checking 0 --param_outlier_removal_use_visibility_confidence 0
```

The description of the full parameter list can be obtained by using -h or from the technical report.