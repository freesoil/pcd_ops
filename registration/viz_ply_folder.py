import open3d as o3d
import os
import argparse
import numpy as np
import re

# Global state variables
pcd_list = []        # List of original point clouds loaded from files
mesh_list = []       # List of computed meshes (one per point cloud)
separated = False    # Flag for point cloud separation
colored = False      # Flag for unique coloring
meshed = False       # Flag for mesh view (vs point clouds)
background_dark = True  # Current background mode (start with dark)
original_points = []    # To store original points for each point cloud
original_colors = []    # To store original colors for each point cloud

def compute_mesh_from_pcd(pcd):
    """Compute a mesh from a point cloud using ball pivoting."""
    pcd.estimate_normals()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    radii = [radius, radius * 2]
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
    except Exception as e:
        print("Mesh generation failed:", e)
        mesh = None
    return mesh

def toggle_background(vis):
    """Toggle the background color between dark and light."""
    global background_dark
    render_option = vis.get_render_option()
    if background_dark:
        render_option.background_color = np.asarray([1, 1, 1])
        print("Background set to light.")
    else:
        render_option.background_color = np.asarray([0, 0, 0])
        print("Background set to dark.")
    background_dark = not background_dark
    return False

def toggle_separation(vis):
    """Toggle separation of the point clouds by translating them along X."""
    global separated, pcd_list, original_points
    if not separated:
        original_points = []
        # Compute an average diagonal length for an appropriate offset
        diags = []
        for pcd in pcd_list:
            bbox = pcd.get_axis_aligned_bounding_box()
            diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
            diags.append(diag)
        offset_value = np.mean(diags) * 0.5 if diags else 1.0

        for i, pcd in enumerate(pcd_list):
            pts = np.asarray(pcd.points)
            original_points.append(pts.copy())
            offset = np.array([(i - (len(pcd_list)-1)/2.0) * offset_value, 0, 0])
            pts = pts + offset
            pcd.points = o3d.utility.Vector3dVector(pts)
            vis.update_geometry(pcd)
        separated = True
        print("Point clouds separated.")
    else:
        # Revert to original positions
        for i, pcd in enumerate(pcd_list):
            pcd.points = o3d.utility.Vector3dVector(original_points[i])
            vis.update_geometry(pcd)
        separated = False
        print("Point clouds recombined.")
    vis.poll_events()
    vis.update_renderer()
    return False

def toggle_coloring(vis):
    """Toggle unique coloring for each point cloud."""
    global colored, pcd_list, original_colors
    # Predefined list of colors (RGB values in range [0,1])
    color_list = [
        [1, 0, 0],   # red
        [0, 1, 0],   # green
        [0, 0, 1],   # blue
        [1, 1, 0],   # yellow
        [1, 0, 1],   # magenta
        [0, 1, 1],   # cyan
    ]
    if not colored:
        original_colors = []
        for i, pcd in enumerate(pcd_list):
            pts_colors = np.asarray(pcd.colors)
            # Save original colors (if available)
            original_colors.append(pts_colors.copy() if pts_colors.size != 0 else None)
            color = color_list[i % len(color_list)]
            num_points = np.asarray(pcd.points).shape[0]
            colors = np.tile(np.array(color), (num_points, 1))
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd)
        colored = True
        print("Point clouds colored with unique colors.")
    else:
        for i, pcd in enumerate(pcd_list):
            if original_colors[i] is not None:
                pcd.colors = o3d.utility.Vector3dVector(original_colors[i])
            else:
                pcd.colors = o3d.utility.Vector3dVector([])
            vis.update_geometry(pcd)
        colored = False
        print("Point clouds reverted to original colors.")
    vis.poll_events()
    vis.update_renderer()
    return False

def toggle_mesh(vis):
    """
    Toggle mesh view.
    When enabled, a mesh is computed for each point cloud, replacing the original point clouds.
    Toggling back removes the meshes and restores the point clouds.
    """
    global meshed, pcd_list, mesh_list
    if not meshed:
        mesh_list = []
        for pcd in pcd_list:
            mesh = compute_mesh_from_pcd(pcd)
            if mesh is not None:
                mesh_list.append(mesh)
                vis.add_geometry(mesh)
                vis.remove_geometry(pcd, reset_bounding_box=False)
        meshed = True
        print("Mesh view enabled. (Press 'M' to toggle back to point clouds.)")
    else:
        for mesh in mesh_list:
            vis.remove_geometry(mesh, reset_bounding_box=False)
        for pcd in pcd_list:
            vis.add_geometry(pcd)
        meshed = False
        print("Mesh view disabled. (Point clouds restored.)")
    vis.poll_events()
    vis.update_renderer()
    return False

def extract_frame_id(filename):
    """
    Extract the frame id as an integer from the file name.
    Assumes the frame id is the trailing digits before the .ply extension.
    e.g. 'object_00123.ply' -> 123
    """
    name, ext = os.path.splitext(filename)
    # Use regex to find trailing digits
    match = re.search(r'(\d+)', name)
    if match:
        return int(match.group(1))
    else:
        return None

def load_and_visualize_ply_folder(folder_path, start_frame, end_frame):
    global pcd_list
    # Get all .ply files in the folder
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".ply")]
    
    # Extract frame ids and filter out files without a valid frame id
    file_frame_pairs = []
    for file in all_files:
        frame_id = extract_frame_id(file)
        if frame_id is not None:
            file_frame_pairs.append((frame_id, file))
    
    if not file_frame_pairs:
        print("No valid PLY files with frame ids found in the specified folder.")
        return

    # Sort by frame id
    file_frame_pairs.sort(key=lambda x: x[0])
    
    # Filter files by specified frame range (inclusive)
    filtered_pairs = []
    for frame_id, file in file_frame_pairs:
        if frame_id >= start_frame and (end_frame is None or frame_id <= end_frame):
            filtered_pairs.append((frame_id, file))
    
    if not filtered_pairs:
        print("No PLY files found in the specified frame range.")
        return

    # Load point clouds from the filtered files
    for frame_id, file in filtered_pairs:
        full_path = os.path.join(folder_path, file)
        pcd = o3d.io.read_point_cloud(full_path)
        pcd_list.append(pcd)
        print(f"Loaded frame {frame_id} from file: {file}")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="PLY Visualizer", width=800, height=600)
    for pcd in pcd_list:
        vis.add_geometry(pcd)

    print("Hotkeys:")
    print("  B: Toggle background (light/dark)")
    print("  S: Toggle separation of point clouds")
    print("  C: Toggle unique coloring for each point cloud")
    print("  M: Toggle mesh view for point clouds")
    print("\nFrame Selection:")
    print(f"  Loaded frames with IDs between {start_frame} and {end_frame if end_frame is not None else 'max'}.")

    vis.register_key_callback(ord("B"), toggle_background)
    vis.register_key_callback(ord("S"), toggle_separation)
    vis.register_key_callback(ord("C"), toggle_coloring)
    vis.register_key_callback(ord("M"), toggle_mesh)

    vis.run()
    vis.destroy_window()

def load_and_visualize_single_ply(file_path):
    """Load and visualize a single PLY file."""
    global pcd_list
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
        
    if not file_path.endswith(".ply"):
        print(f"Error: {file_path} is not a PLY file.")
        return
        
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        pcd_list.append(pcd)
        print(f"Loaded point cloud from file: {file_path}")
        
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="PLY Visualizer", width=800, height=600)
        vis.add_geometry(pcd)
        
        print("Hotkeys:")
        print("  B: Toggle background (light/dark)")
        print("  S: Toggle separation of point clouds")
        print("  C: Toggle unique coloring for each point cloud")
        print("  M: Toggle mesh view for point clouds")
        
        vis.register_key_callback(ord("B"), toggle_background)
        vis.register_key_callback(ord("S"), toggle_separation)
        vis.register_key_callback(ord("C"), toggle_coloring)
        vis.register_key_callback(ord("M"), toggle_mesh)
        
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"Error loading point cloud: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and visualize PLY point clouds from a folder or individual file."
    )
    parser.add_argument("path", type=str, help="Path to folder containing .ply files or a single .ply file")
    parser.add_argument("--start", type=int, default=0, help="Start frame id (inclusive, for folder mode only)")
    parser.add_argument("--end", type=int, default=None, help="End frame id (inclusive, for folder mode only)")
    
    args = parser.parse_args()
    
    # Check if the path is a file or directory
    if os.path.isfile(args.path):
        load_and_visualize_single_ply(args.path)
    elif os.path.isdir(args.path):
        load_and_visualize_ply_folder(args.path, args.start, args.end)
    else:
        print(f"Error: {args.path} is not a valid file or directory")

