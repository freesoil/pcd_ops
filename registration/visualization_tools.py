import numpy as np
import copy
import open3d as o3d
import argparse


def merge_and_colorize(pcd_list):
    global_pcd = o3d.geometry.PointCloud()
    for i in range(len(pcd_list)):
        pcd = copy.deepcopy(pcd_list[i])
        color = np.random.rand(1, 3)
        colors = np.tile(color, (np.asarray(pcd.points).shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        global_pcd += pcd
    return global_pcd

def visualize_colored_mesh(mesh_path):
    """
    Visualize a PLY file with colored mesh
    Args:
        mesh_path (str): Path to the PLY file
    """
    # Load the mesh from PLY file
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Ensure the mesh has vertex colors
    if not mesh.has_vertex_colors():
        print("Mesh has no vertex colors, applying random colors")
        # Generate random colors for vertices
        vertices = np.asarray(mesh.vertices)
        colors = np.random.uniform(0, 1, size=(len(vertices), 3))
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Compute vertex normals for better visualization
    mesh.compute_vertex_normals()
    
    # Create visualizer instance
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add the mesh to the visualizer
    vis.add_geometry(mesh)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])  # Gray background
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()


def main():
    """
    Main function to run the mesh visualization from command line
    """
    parser = argparse.ArgumentParser(description='Visualize a PLY file with colored mesh')
    parser.add_argument('--mesh_path', type=str, required=True,
                       help='Path to the PLY file to visualize')
    
    args = parser.parse_args()
    visualize_colored_mesh(args.mesh_path)

if __name__ == "__main__":
    main()