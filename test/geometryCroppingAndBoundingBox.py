import stl
from stl import mesh
import vtkplotlib as vpl
import open3d as o3d
from vtkmodules import *
import copy
import numpy as np

import aspose.threed as a3d


def demo_crop_geometry():
    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")

    body = o3d.io.read_point_cloud("../outputs/3d_models/clean.ply")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Full Body", width=800, height=800, left=800, top=200)
    vis.add_geometry(body)
    vis.run()
    vis.destroy_window()
    # o3d.visualization.draw_geometries_with_editing([body])


if __name__ == "__main__":
    demo_crop_geometry()

    # Read the point cloud
    body = o3d.io.read_point_cloud("../outputs/3d_models/clean.ply")
    head = o3d.io.read_point_cloud("../outputs/3d_models/cropp \ ed_1.ply")

    # Visualization of point cloud with window size
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Head", width=800, height=800, left=800, top=200)
    vis.add_geometry(head)
    vis.run()
    vis.capture_screen_image("image-front.png", do_render=True)
    vis.destroy_window()

    # apply final filter poisson
    distances = head.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    print(avg_dist)
    print(radius)

    poisson_mesh = \
    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(head, depth=9, width=0, scale=2, linear_fit=False)[0]
    bbox = head.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)


    def lod_mesh_export(p_mesh_crop, lods, extension, path):
        mesh_lods = {}
        for i in lods:
            mesh_lod = p_mesh_crop.simplify_quadric_decimation(i)
            o3d.io.write_triangle_mesh(path + "head" + extension, mesh_lod)
            mesh_lods[i] = mesh_lod
        print("generation of " + str(i) + " LoD successful")
        return mesh_lods


    output_path = "./"
    my_lods = lod_mesh_export(p_mesh_crop, [len(body.points)], ".ply", output_path)

    # pcd = o3d.io.read_point_cloud("head.ply")
    # print("Testing mesh in open3d ...")
    # mesh = o3d.io.read_triangle_mesh("head.ply")
    # print("Computing normal and rendering it.")
    # mesh.compute_vertex_normals()
    # print(np.asarray(mesh.triangle_normals))
    # o3d.visualization.draw_geometries([mesh],window_name="Without head",
    #                                 width=800,height=800,left=800,top=200)

    # head.estimate_normals(
    # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
    # print(dir(head))

    # o3d.visualization.draw_geometries([head],window_name="head",
    #                                     width = 800, height = 800, left=800, top=200)

    aabb = head.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    obb = head.get_oriented_bounding_box()
    obb.color = (0, 1, 0)

    o3d.visualization.draw_geometries([head, aabb, obb],
                                      window_name="With color head",
                                      width=800, height=800, left=50, top=50)
