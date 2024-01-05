import numpy as np
import open3d as o3d
import copy


def landmarksEstimation(file_path):
    body = o3d.io.read_point_cloud(file_path)

    # front body display
    mesh_f = copy.deepcopy(body)
    R1 = body.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
    mesh_f.rotate(R1, center=(0, 0, 0))

    # back body display
    mesh_b = copy.deepcopy(body)
    R2 = body.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-90)))
    mesh_b.rotate(R2, center=(0, 0, 0))

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=800, height=800, left=800, top=200)
    vis.add_geometry(mesh_f)
    vis.run()  # user picks points
    vis.destroy_window()
    vis.get_picked_points()

    pointcloud_as_array = np.asarray(body.points)
    points = pointcloud_as_array[vis.get_picked_points()]
    pointsall = points[:, :]
    print(f"Pointall: {pointsall}")

    with open("outputs/text_files/Frontpoints.txt", mode='w') as f:  # I add the mode='w'
        for i in range(len(pointsall)):
            f.write("%f    " % float(pointsall[i][0].item()))
            f.write("%f    " % float(pointsall[i][1].item()))
            f.write("%f    \n" % float(pointsall[i][2].item()))

    data1 = np.loadtxt("outputs/text_files/Frontpoints.txt")
    pcdext = o3d.io.read_point_cloud("outputs/text_files/Frontpoints.txt", format="xyz")
    o3d.io.write_point_cloud("outputs/3d_models/Frontpoints.ply", pcdext)
    mesh = o3d.io.read_point_cloud("outputs/3d_models/Frontpoints.ply")
    array = np.asarray(mesh.points)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=800, height=800, left=800, top=200)
    vis.add_geometry(mesh_b)
    vis.run()  # user picks points
    vis.destroy_window()
    vis.get_picked_points()

    pointcloud_as_array1 = np.asarray(body.points)
    points1 = pointcloud_as_array1[vis.get_picked_points()]

    pointsall1 = points1[:, :]

    print(f"Pointall: {pointsall1}")

    with open("outputs/text_files/Backpoints.txt", mode='w') as f:  # I add the mode='w'
        for i in range(len(pointsall1)):
            f.write("%f    " % float(pointsall1[i][0].item()))
            f.write("%f    " % float(pointsall1[i][1].item()))
            f.write("%f    \n" % float(pointsall1[i][2].item()))

    data2 = np.loadtxt("outputs/text_files/Backpoints.txt")
    pcdext1 = o3d.io.read_point_cloud("outputs/text_files/Backpoints.txt", format="xyz")
    o3d.io.write_point_cloud("outputs/3d_models/Backpoints.ply", pcdext1)
    mesh1 = o3d.io.read_point_cloud("outputs/3d_models/Backpoints.ply")
    array1 = np.asarray(mesh1.points)

    def create_geometry_at_points(pointsfinal):
        geometries = o3d.geometry.TriangleMesh()
        for array in pointsfinal:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)  # create a small sphere to represent point
            sphere.translate(array)  # translate this sphere to point
            geometries += sphere
        geometries.paint_uniform_color([1.0, 0.0, 1.0])
        return geometries


    highlight_pnts = create_geometry_at_points(mesh.points)
    highlight_pnts1 = create_geometry_at_points(mesh1.points)

    o3d.visualization.draw_geometries([body, highlight_pnts, highlight_pnts1], width=800, height=900, left=1050,
                                      top=100)
