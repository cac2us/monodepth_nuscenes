import open3d as o3d

def visualize_pcd(pcd_file_path):
    # Step 1: Load the .pcd file
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    import pdb; pdb.set_trace()
    if not pcd.has_points():
        print("The point cloud is empty.")
        return

    # Step 2: Visualize the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 3.0
    num_points = len(pcd.points)
    black_color = [0, 0, 0]  # RGB values for black (all 0)
    pcd.colors = o3d.utility.Vector3dVector([black_color] * num_points)
    o3d.visualization.draw_geometries([pcd])

# Example usage:
input_pcd_file = "./output.pcd"
visualize_pcd(input_pcd_file)
