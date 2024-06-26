import open3d as o3d
import numpy as np
import copy


def loadAndShape(file):
    # Read the point cloud
    pcd = o3d.io.read_point_cloud(file)

    # rotate to the Front side
    mesh_r = copy.deepcopy(pcd)
    R = pcd.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
    mesh_r.rotate(R, center=(0, 0, 0))

    # Visualization of point cloud with window size
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Front", width=800, height=800, left=800, top=200)
    vis.add_geometry(mesh_r)
    vis.run()
    vis.destroy_window()

    # Remove the table
    # Define the cropping box of the table
    min_bound = np.array([50.5, -2713.3, 50.5])
    max_bound = np.array([1671.3, -2070.3, 2513.3])

    # Filter points within the bounding box
    filtered_indices = np.all((mesh_r.points >= min_bound) & (mesh_r.points <= max_bound), axis=1)
    cropped_point_cloud = mesh_r.select_by_index(np.where(~filtered_indices)[0])

    # save to .ply
    # o3d.io.write_point_cloud("removetable.ply", cropped_point_cloud)
    # mesh = o3d.io.read_point_cloud("removetable.ply")
    # o3d.visualization.draw_geometries([cropped_point_cloud],window_name="Without Table",width=800,height=800,left=50,top=50)
    # ---------------------------------------------------------------------------------------------

    # remove the outlier by index first time

    voxel_size = 0.05
    pcd_downsampled = cropped_point_cloud.voxel_down_sample(voxel_size=voxel_size)
    uni_down_pcd = pcd_downsampled.uniform_down_sample(every_k_points=4)

    # o3d.visualization.draw_geometries([uni_down_pcd],width=800,height=800,left=50,top=50)

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        # outlier_cloud.paint_uniform_color([1, 0, 0])
        # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        # o3d.visualization.draw_geometries([inlier_cloud],width=800,height=800,left=50,top=50)

    print("Statistical oulier removal")
    cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    display_inlier_outlier(uni_down_pcd, ind)

    # apply final filter poisson
    distances = cl.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    print(avg_dist)
    print(radius)

    poisson_mesh = \
    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cl, depth=9, width=0, scale=2, linear_fit=False)[0]
    bbox = cl.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    def lod_mesh_export(p_mesh_crop, lods, extension, path):
        mesh_lods = {}
        for i in lods:
            mesh_lod = p_mesh_crop.simplify_quadric_decimation(i)
            o3d.io.write_triangle_mesh(path + "clean" + extension, mesh_lod)
            mesh_lods[i] = mesh_lod
        print("generation of " + str(i) + " LoD successful")
        return mesh_lods

    output_path = "outputs/3d_models/"
    my_lods = lod_mesh_export(p_mesh_crop, [len(pcd.points)], ".ply", output_path)

    pcd = o3d.io.read_point_cloud("outputs/3d_models/clean.ply")
    o3d.visualization.draw_geometries([pcd], window_name="Body Shape",
                                      width=800, height=800, left=50, top=50)
