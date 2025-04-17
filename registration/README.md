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

To run the registration pipeline, execute the script demo.py.

```bash
python demo.py
```

In this script you can modify the input folder and some parameters. The combined (registered) point cloud and the isolated registered point clouds will be exported to the folder data/results/<name>. Notice that just the point clouds that presented an acceptable registration will be considered.

The proposed approach relies on certain assumptions about the input data: each point cloud should contain color information and be part of a sequential acquisition, ensuring substantial overlap with preceding scans. This sequence order is represented by the corresponding positions of an input list of point clouds. 


