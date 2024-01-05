import numpy as np
import copy
import open3d as o3d
from scipy.spatial import distance
from tkinter import messagebox
import base64
import requests
import json

def lod_mesh_export(p_mesh_crop, lods, extension, path):
    mesh_lods={}
    for i in lods:
        mesh_lod = p_mesh_crop.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"cleanBody"+extension, mesh_lod)
        mesh_lods[i]=mesh_lod
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods


def Clean3D(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    print(np.asarray(pcd))
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    print(avg_dist)
    print(radius)
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0, scale=2, linear_fit=False)[0]
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)
    output_path="./"
    lod_mesh_export(p_mesh_crop, [len(pcd.points)], ".ply", output_path)
    # mesh = o3d.io.read_point_cloud("Clean body.ply")
    # o3d.visualization.draw_geometries([mesh])


def SelectedPoints(ply_file):
    body = o3d.io.read_point_cloud(ply_file)
    # Rotate the point cloud
    # mesh_r = copy.deepcopy(body)
    # R = body.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))  # Define rotation matrix
    # mesh_r.rotate(R, center=(0, 0, 0))


    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Front", width=800, height=800, left=800, top=200)
    vis.add_geometry(body)
    vis.run()  # user picks points
    # vis.capture_screen_image("file1.png", do_render=True)
    vis.destroy_window()
    vis.get_picked_points()

    pointcloud_as_array = np.asarray(body.points)
    points = pointcloud_as_array[vis.get_picked_points()]
    pointsall = points[:, :]

    with open("markersFullSelected.txt", mode='a') as f:
        for i in range(len(pointsall)):
            f.write("%f    " % float(pointsall[i][0].item()))
            f.write("%f    " % float(pointsall[i][1].item()))
            f.write("%f    \n" % float(pointsall[i][2].item()))

    dist1 = distance.euclidean(pointsall[0, :], pointsall[1, :])
    dist2 = distance.euclidean(pointsall[1, :], pointsall[2, :])
    dist3 = distance.euclidean(pointsall[0, :], pointsall[2, :])

    print(f"Distance 1 in cm: {dist1 * 0.1}")
    print(f"Distance 2 in cm: {dist2 * 0.1}")
    print(f"Distance 3 in cm: {dist3 * 0.1}")



    # Read the saved markers
    pcdText = o3d.io.read_point_cloud("markersFullSelected.txt", format="xyz")
    o3d.io.write_point_cloud("selected_points.ply", pcdText)
    pcdFull = o3d.io.read_point_cloud("selected_points.ply")


    # Create geometries for markers points
    markers_points = create_geometry_at_points(pcdFull.points)

    # Visualize the body with landmarks
    o3d.visualization.draw_geometries([body, markers_points], window_name="Full Body with landmarks", width=800,
                                      height=800, left=800, top=200)

def create_geometry_at_points(pointcloud_as_array2):
    geometries = o3d.geometry.TriangleMesh()
    for array in pointcloud_as_array2:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20) #create a small sphere to represent point
        sphere.translate(array) #translate this sphere to point
        geometries += sphere
    geometries.paint_uniform_color([1, 0.8, 0.0])
    return geometries

def distances(self):
    if not self.file_path:
        messagebox.showerror("Error", "Please select a file")
    else:
        if not self.Clean3DPLY:
            messagebox.showerror("Error", "Cleaned File Doesn't Exist")
        else:
            dists = GetDistances(self.Clean3DPLY)

            # Initialize an empty string to store the text
            text = ""

            # Round the distances and store them in variables
            dist1 = round(dists[0])
            dist2 = round(dists[1])
            dist12 = round(dists[2])
            dist3 = round(dists[3])
            dist4 = round(dists[4])
            dist34 = round(dists[5])
            dist5 = round(dists[6])
            dist6 = round(dists[7])
            dist7 = round(dists[8])
            dist67 = round(dists[9])
            dist8 = round(dists[10])
            dist9 = round(dists[11])
            dist89 = round(dists[12])
            dist10 = round(dists[13])
            dist11 = round(dists[14])

            # Create a text representation of the distances
            text = (
                "Left upper Arm: " + str(dist1) + " cm" + "\n" +
                "Left Forearm: " + str(dist2) + " cm" + "\n" +
                "Left Arm: " + str(dist12) + " cm" + "\n" +
                "Right upper Arm: " + str(dist3) + " cm" + "\n" +
                "Right Forearm: " + str(dist4) + " cm" + "\n" +
                "Right Arm: " + str(dist34) + " cm" + "\n" +
                "Upper back body: " + str(dist5) + " cm" + "\n" +
                "Right Thigh: " + str(dist6) + " cm" + "\n" +
                "Right Shin: " + str(dist7) + " cm" + "\n" +
                "Right Leg: " + str(dist67) + " cm" + "\n" +
                "Left Thigh: " + str(dist8) + " cm" + "\n" +
                "Left Shin: " + str(dist9) + " cm" + "\n" +
                "Left Leg: " + str(dist89) + " cm" + "\n" +
                "Full Tall: " + str(dist10) + " cm" + "\n"
            )

            # Update the text in the guide label
            self.guide.config(text=text)


def Markers_Points(ply_file):
    pointcloud1 = o3d.io.read_point_cloud(ply_file)
    pointcloud_as_array2 = np.asarray(pointcloud1.points)

    data1 = np.loadtxt("selected_points.txt")
    pcdext = o3d.io.read_point_cloud("selected_points.txt", format="xyz")

    o3d.io.write_point_cloud("selected_points.ply", pcdext)
    mesh = o3d.io.read_point_cloud("selected_points.ply")
    array=np.asarray(mesh.points)
    pointsall=array[:,:]
    highlight_pnts = create_geometry_at_points(mesh.points)
    lines = [[0, 5],[1, 4],[0, 7],[1, 8],[9, 10],[11, 12],[12, 13],[13, 14],[14, 15],[15, 16],[16, 17],[2, 24],[3, 25],[23,27],[22,26], [17,31],[31,30],[30,6],]
    points=o3d.utility.Vector3dVector(pointsall)
    lines=o3d.utility.Vector2iVector(lines)
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(points,lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pointcloud1,highlight_pnts,line_set])


def GetDistances():

    landmarks = np.loadtxt("Frontpoints.txt")
    pointsall=landmarks[:,:]
    point_1=pointsall[0,:]
    X1=point_1[0]
    Y1=point_1[1]
    Z1=point_1[2]
    point_2=pointsall[1,:]
    X2=point_2[0]
    Y2=point_2[1]
    Z2=point_2[2]
    point_3=pointsall[2,:]
    X3=point_3[0]
    Y3=point_3[1]
    Z3=point_3[2]
    point_4=pointsall[3,:]
    X4=point_4[0]
    Y4=point_4[1]
    Z4=point_4[2]
    point_5=pointsall[4,:]
    X5=point_5[0]
    Y5=point_5[1]
    Z5=point_5[2]
    point_6=pointsall[5,:]
    X6=point_6[0]
    Y6=point_6[1]
    Z6=point_6[2]
    point_7=pointsall[6,:]
    X7=point_7[0]
    Y7=point_7[1]
    Z7=point_7[2]
    point_8=pointsall[7,:]
    X8=point_8[0]
    Y8=point_8[1]
    Z8=point_8[2]
    point_9=pointsall[8,:]
    X9=point_9[0]
    Y9=point_9[1]
    Z9=point_9[2]
    point_10=pointsall[9,:]
    X10=point_10[0]
    Y10=point_10[1]
    Z10=point_10[2]
    point_11=pointsall[10,:]
    X11=point_11[0]
    Y11=point_11[1]
    Z11=point_11[2]
    point_12=pointsall[11,:]
    X12=point_12[0]
    Y12=point_12[1]
    Z12=point_12[2]
    point_13=pointsall[12,:]
    X13=point_13[0]
    Y13=point_13[1]
    Z13=point_13[2]
    point_14=pointsall[13,:]
    X14=point_14[0]
    Y14=point_14[1]
    Z14=point_14[2]
    point_15=pointsall[14,:]
    X15=point_15[0]
    Y15=point_15[1]
    Z15=point_15[2]
    point_16=pointsall[15,:]
    X16=point_16[0]
    Y16=point_16[1]
    Z16=point_16[2]
    point_17=pointsall[16,:]
    X17=point_17[0]
    Y17=point_17[1]
    Z17=point_17[2]

    point_25=pointsall[17,:]
    X25=point_25[0]
    Y25=point_25[1]
    Z25=point_25[2]

    point_26=pointsall[18,:]
    X26=point_26[0]
    Y26=point_26[1]
    Z26=point_26[2]

    point_27=pointsall[19,:]
    X27=point_27[0]
    Y27=point_27[1]
    Z27=point_27[2]
    
    # point_28=pointsall[20,:]
    # X28=point_28[0]
    # Y28=point_28[1]
    # Z28=point_28[2]

    # point_29=pointsall[20,:]
    # X29=point_29[0]
    # Y29=point_29[1]
    # Z29=point_29[2]

    #point_34=pointsall[33,:]
    #X34=point_34[0]
    #Y34=point_34[1]
    #Z34=point_34[2]

    #point_35=pointsall[34,:]
    #X35=point_35[0]
    #Y35=point_35[1]
    #Z35=point_35[2]


    dist1=distance.euclidean(point_1,point_6)
    dist2=distance.euclidean(point_1,point_8)
    dist12=distance.euclidean(point_6,point_8)

    dist3=distance.euclidean(point_2,point_5)
    dist4=distance.euclidean(point_2,point_9)
    dist34=distance.euclidean(point_5,point_9)

    dist5=distance.euclidean(point_7,point_12)

    dist6=distance.euclidean(point_4,point_26)
    dist7=distance.euclidean(point_4,point_27)
    dist67=distance.euclidean(point_26,point_27)

    dist8=distance.euclidean(point_3,point_25)
    dist9=distance.euclidean(point_3,point_27)
    dist89=distance.euclidean(point_25,point_27)

    dist10=distance.euclidean(point_10,point_11)
    dist11=distance.euclidean(point_11,point_17)

    return [dist1*0.1,dist2*0.1,dist12*0.1,dist3*0.1,dist4*0.1,dist34*0.1,dist5*0.1,dist6*0.1,dist7*0.1,dist67*0.1,dist8*0.1,dist9*0.1,dist89*0.1,dist10*0.1,dist11*0.1]


def PLYtoSTL(output_name,ply_file):
    mesh = o3d.io.read_triangle_mesh(ply_file)
    #rotate 180 degrees around x axis
    mesh.rotate(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), center=(0, 0, 0))
    mesh = o3d.geometry.TriangleMesh.compute_triangle_normals (mesh)
    o3d.io.write_triangle_mesh(output_name, mesh)

def Final_Result(ply_file):
    pointcloud1 = o3d.io.read_point_cloud(ply_file)
    pointcloud_as_array2 = np.asarray(pointcloud1.points)

    #data1 = np.loadtxt("selected_points.txt")
    #pcdext = o3d.io.read_point_cloud("selected_points.txt", format="xyz")

    #o3d.io.write_point_cloud("selected_points.ply", pcdext)
    mesh = o3d.io.read_point_cloud("selected_points.ply")
    array=np.asarray(mesh.points)
    pointsall=array[:,:]
    highlight_pnts = create_geometry_at_points(mesh.points)
    lines = [[0, 5],[1, 4],[0, 7],[1, 8],[9, 10],[11, 12],[12, 13],[13, 14],[14, 15],[15, 16],[16, 17],[2, 24],[3, 25],[23,27],[22,26], [17,31],[31,30],[30,6],]
    points=o3d.utility.Vector3dVector(pointsall)
    lines=o3d.utility.Vector2iVector(lines)
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(points,lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([highlight_pnts,line_set])

