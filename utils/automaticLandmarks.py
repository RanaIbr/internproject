import open3d as o3d
import numpy as np
import copy
from scipy.spatial import distance

full_markers_file = "../outputs/text_files/markersFullAutomatic.txt"  # File to store marker points
# body = o3d.io.read_point_cloud("Dr Mohamad.ply")  # Read the point cloud data

# Clear the contents of the markersFull.txt file
with open(full_markers_file, mode='w') as f:
    pass


def getDistances():
    landmarks = np.loadtxt(full_markers_file)
    pointsall = landmarks[:, :]
    print(pointsall)
    point_1 = pointsall[0, :]
    point_3 = pointsall[2, :]
    point_4 = pointsall[3, :]
    point_5 = pointsall[4, :]
    point_7 = pointsall[6, :]
    point_8 = pointsall[7, :]
    point_14 = pointsall[13, :]
    point_17 = pointsall[16, :]
    point_18 = pointsall[17, :]
    point_19 = pointsall[18, :]
    point_20 = pointsall[19, :]
    point_25 = pointsall[24, :]
    point_26 = pointsall[25, :]
    point_27 = pointsall[26, :]

    dist1 = distance.euclidean(point_4, point_7)
    dist2 = distance.euclidean(point_7, point_17)
    dist3 = distance.euclidean(point_4, point_17)
    dist4 = distance.euclidean(point_5, point_8)
    dist5 = distance.euclidean(point_8, point_18)
    dist6 = distance.euclidean(point_5, point_18)
    dist7 = distance.euclidean(point_3, point_14)
    dist8 = distance.euclidean(point_4, point_26)
    dist9 = distance.euclidean(point_20, point_27)
    dist10 = distance.euclidean(point_26, point_27)
    dist11 = distance.euclidean(point_19, point_25)
    dist12 = distance.euclidean(point_19, point_25)
    dist13 = distance.euclidean(point_25, point_27)
    dist14 = distance.euclidean(point_1, point_25)

    return [dist1 * 0.1, dist2 * 0.1, dist3 * 0.1, dist4 * 0.1, dist5 * 0.1, dist6 * 0.1, dist7 * 0.1, dist8 * 0.1,
            dist9 * 0.1, dist10 * 0.1, dist11 * 0.1, dist12 * 0.1, dist13 * 0.1, dist14 * 0.1]


def create_geometry_at_points(array_):
    # Create geometry at given points in the form of small spheres.
    geometries = o3d.geometry.TriangleMesh()
    for array_ in array_:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)  # Create a small sphere to represent a point
        sphere.translate(array_)  # Translate this sphere to the given point
        geometries += sphere
    geometries.paint_uniform_color([1, 0.8, 0.0])  # Set the color of the spheres
    return geometries


def slicing_lower(lines, start, end):
    points = lines[start:end, :]
    data1 = np.asarray(points)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(data1)

    yMax = max(pcd1.points, key=lambda x: x[1])
    zMax = max(pcd1.points, key=lambda x: x[2])
    landmark_points = [zMax, yMax]

    landmark_points = np.asarray(landmark_points)

    with open(full_markers_file, mode='a') as f:
        for i in range(len(landmark_points)):
            f.write("%f    " % float(landmark_points[i][0].item()))
            f.write("%f    " % float(landmark_points[i][1].item()))
            f.write("%f    \n" % float(landmark_points[i][2].item()))


def slicing(lines, start, end):
    points = lines[start:end, :]
    data1 = np.asarray(points)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(data1)

    zMin = min(pcd1.points, key=lambda x: x[2])
    landmark_points = [zMin]

    landmark_points = np.asarray(landmark_points)

    with open(full_markers_file, mode='a') as f:
        for i in range(len(landmark_points)):
            f.write("%f    " % float(landmark_points[i][0].item()))
            f.write("%f    " % float(landmark_points[i][1].item()))
            f.write("%f    \n" % float(landmark_points[i][2].item()))


def slicing_shoulders(lines, start, end):
    points = lines[start:end, :]
    data1 = np.asarray(points)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(data1)

    xMax = max(pcd1.points, key=lambda x: x[0])
    xMin = min(pcd1.points, key=lambda x: x[0])
    zMax = max(pcd1.points, key=lambda x: x[2])
    landmark_points = [xMax, xMin, zMax]

    landmark_points = np.asarray(landmark_points)

    with open(full_markers_file, mode='a') as f:
        for i in range(len(landmark_points)):
            f.write("%f    " % float(landmark_points[i][0].item()))
            f.write("%f    " % float(landmark_points[i][1].item()))
            f.write("%f    \n" % float(landmark_points[i][2].item()))


def slicing_hands(lines, start, end):
    points = lines[start:end, :]
    data1 = np.asarray(points)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(data1)

    xMax = max(pcd1.points, key=lambda x: x[0])
    xMin = min(pcd1.points, key=lambda x: x[0])
    landmark_points = [xMax, xMin]

    landmark_points = np.asarray(landmark_points)

    with open(full_markers_file, mode='a') as f:
        for i in range(len(landmark_points)):
            f.write("%f    " % float(landmark_points[i][0].item()))
            f.write("%f    " % float(landmark_points[i][1].item()))
            f.write("%f    \n" % float(landmark_points[i][2].item()))


def slicing_elbows(lines, start, end):
    points = lines[start:end, :]
    data1 = np.asarray(points)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(data1)

    xMax = max(pcd1.points, key=lambda x: x[0])
    xMin = min(pcd1.points, key=lambda x: x[0])

    landmark_points = [xMax, xMin]

    landmark_points = np.asarray(landmark_points)

    with open(full_markers_file, mode='a') as f:
        for i in range(len(landmark_points)):
            f.write("%f    " % float(landmark_points[i][0].item()))
            f.write("%f    " % float(landmark_points[i][1].item()))
            f.write("%f    \n" % float(landmark_points[i][2].item()))


def slicing_feets(lines, start, end):
    points = lines[start:end, :]
    data1 = np.asarray(points)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(data1)

    yMin = min(pcd1.points, key=lambda x: x[1])
    zMin = min(pcd1.points, key=lambda x: x[2])
    landmark_points = [zMin, yMin]

    landmark_points = np.asarray(landmark_points)

    with open(full_markers_file, mode='a') as f:
        for i in range(len(landmark_points)):
            f.write("%f    " % float(landmark_points[i][0].item()))
            f.write("%f    " % float(landmark_points[i][1].item()))
            f.write("%f    \n" % float(landmark_points[i][2].item()))


def slicing_front_feets(lines, start, end):
    points = lines[start:end, :]
    data1 = np.asarray(points)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(data1)

    xMax = max(pcd1.points, key=lambda x: x[0])
    zMax = max(pcd1.points, key=lambda x: x[2])
    landmark_points = [xMax, zMax]

    landmark_points = np.asarray(landmark_points)

    with open(full_markers_file, mode='a') as f:
        for i in range(len(landmark_points)):
            f.write("%f    " % float(landmark_points[i][0].item()))
            f.write("%f    " % float(landmark_points[i][1].item()))
            f.write("%f    \n" % float(landmark_points[i][2].item()))


def slicing_head(lines, start, end):
    # Process first set of points
    points = lines[start:end, :]
    data = np.asarray(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    yMax = max(pcd.points, key=lambda x: x[1])  # Maximum y-coordinate
    zMax = max(pcd.points, key=lambda x: x[2])  # Maximum z-coordinate
    zMin = min(pcd.points, key=lambda x: x[2])  # Minimum z-coordinate

    landmark_points = [yMax, zMax, zMin]  # Calculate landmark points based on coordinate values

    landmark_points = np.asarray(landmark_points)

    with open(full_markers_file, mode='a') as f:
        for i in range(len(landmark_points)):
            f.write("%f    " % float(landmark_points[i][0].item()))
            f.write("%f    " % float(landmark_points[i][1].item()))
            f.write("%f    \n" % float(landmark_points[i][2].item()))


def startup(clean):
    body = o3d.io.read_point_cloud(clean)  # Read the point cloud data
    mesh_array = np.asarray(body.points)
    colors = np.asarray(body.colors)

    # Rotate the point cloud
    mesh_r = copy.deepcopy(body)
    R = body.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))  # Define rotation matrix
    mesh_r.rotate(R, center=(0, 0, 0))

    # Visualize the point cloud
    o3d.visualization.draw_geometries([mesh_r], window_name="Front", width=800, height=800, left=800,
                                      top=200)

    # Normal estimation
    mesh_r.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    lines = np.asarray(mesh_r.points)

    slicing_head(lines, 0, 6000)
    slicing_shoulders(lines, 6500, 7100)
    slicing_elbows(lines, 20000, 21000)
    slicing(lines, 20000, 21000)
    slicing(lines, 22000, 23000)
    slicing(lines, 24000, 26000)
    slicing(lines, 30000, 31000)
    slicing(lines, 32000, 33000)
    slicing(lines, 34000, 35000)
    slicing(lines, 36000, 38000)
    slicing(lines, 39000, 40000)
    slicing_hands(lines, 40000, 46000)
    slicing_lower(lines, 58000, 60000)
    slicing_feets(lines, 65800, 66000)
    slicing_shoulders(lines, 70000, len(lines))
    slicing_front_feets(lines, 69500, 69550)

    # Rotate the body
    mesh_r = copy.deepcopy(body)
    R = body.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
    mesh_r.rotate(R, center=(0, 0, 0))

    # Read the saved markers
    pcdText = o3d.io.read_point_cloud(full_markers_file, format="xyz")
    o3d.io.write_point_cloud("outputs/3d_models/markersFullAutomatic.ply", pcdText)
    pcdFull = o3d.io.read_point_cloud("outputs/3d_models/markersFullAutomatic.ply")

    # Create geometries for markers points
    markers_points = create_geometry_at_points(pcdFull.points)

    # Visualize the body with landmarks
    o3d.visualization.draw_geometries([mesh_r, markers_points], window_name="Full Body with landmarks", width=800,
                                      height=800, left=800, top=200)

    dists = getDistances()
    dist1 = round(dists[0])
    dist2 = round(dists[1])
    dist3 = round(dists[2])
    dist4 = round(dists[3])
    dist5 = round(dists[4])
    dist6 = round(dists[5])
    dist7 = round(dists[6])
    dist8 = round(dists[7])
    dist9 = round(dists[8])
    dist10 = round(dists[9])
    dist11 = round(dists[10])
    dist12 = round(dists[11])
    dist13 = round(dists[12])
    dist14 = round(dists[13])

    for x in dists:
        text = "Left upper Arm: " + str(dist1) + " cm" + "\n" + "Left Forearm: " + str(
            dist2) + " cm" + "\n" + "Left Arm: " + str(dist3) + " cm" + "\n" + "Right upper Arm: " + str(
            dist4) + " cm" + "\n" + "Right Forearm: " + str(dist5) + " cm" + "\n" + "Right Arm: " + str(
            dist6) + " cm" + "\n" + "Upper back body: " + str(
            dist7) + " cm" + "\n" + "Right Thigh: " + str(dist8) + " cm" + "\n" + "Right Shin: " + str(
            dist9) + " cm" + "\n" + "Right Leg: " + str(dist10) + " cm" + "\n" + "Left Thigh: " + str(
            dist11) + " cm" + "\n" + "Left Shin: " + str(dist12) + " cm" + "\n" + "Left Leg: " + str(
            dist13) + " cm" + "\n" + "Full Tall: " + str(dist14) + " cm" + "\n"
    print(text)

