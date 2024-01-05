import numpy as np
import open3d as o3d
import copy
from scipy.spatial import distance
import math
import pygetwindow as gw
import pyautogui


def frontLandmarks(file_path):
    body = o3d.io.read_point_cloud(file_path)

    # front body display
    mesh_f = copy.deepcopy(body)
    R1 = body.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
    mesh_f.rotate(R1, center=(0, 0, 0))

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

    point_1 = pointsall[0, :]
    point_2 = pointsall[1, :]
    point_3 = pointsall[2, :]
    point_4 = pointsall[3, :]
    point_5 = pointsall[4, :]
    point_6 = pointsall[5, :]
    point_7 = pointsall[6, :]
    point_8 = pointsall[7, :]
    point_9 = pointsall[8, :]
    point_10 = pointsall[9, :]
    point_11 = pointsall[10, :]
    point_12 = pointsall[11, :]
    point_13 = pointsall[12, :]

    z_max = max(body.points, key=lambda x: x[2])
    z_min = min(body.points, key=lambda x: x[2])
    print(z_max)
    print(z_min)

    dist1 = (distance.euclidean(point_1, point_2)) * 0.1
    dist1 = round(dist1, 2)
    dist2 = (distance.euclidean(point_1, point_3)) * 0.1
    dist2 = round(dist2, 2)
    dist3 = (distance.euclidean(point_2, point_4)) * 0.1
    dist3 = round(dist3, 2)
    dist4 = (distance.euclidean(point_3, point_5)) * 0.1
    dist4 = round(dist4, 2)
    dist5 = (distance.euclidean(point_4, point_6)) * 0.1
    dist5 = round(dist5, 2)
    dist6 = (distance.euclidean(point_7, point_9)) * 0.1
    dist6 = round(dist6, 2)
    dist7 = (distance.euclidean(point_8, point_10)) * 0.1
    dist7 = round(dist7, 2)
    dist8 = (distance.euclidean(point_9, point_11)) * 0.1
    dist8 = round(dist8, 2)
    dist9 = (distance.euclidean(point_10, point_12)) * 0.1
    dist9 = round(dist9, 2)
    dist10 = (distance.euclidean(z_max, z_min)) * 0.1
    dist10 = round(dist10, 2)

    dist11 = (distance.euclidean(point_13, z_min)) * 0.1
    dist11 = round(dist11, 2)

    dist12 = (distance.euclidean(point_13, z_max)) * 0.1
    dist12 = round(dist12, 2)

    ratio12 = dist11 / dist12
    ratio12 = round(ratio12, 2)

    def calculateAngle(landmark1, landmark2, landmark3):
        x1, y1, _ = landmark1
        x2, y2, _ = landmark2
        x3, y3, _ = landmark3

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        if angle < 0:
            angle += 360
        return angle

    left_elbow_angle = calculateAngle(point_7, point_9, point_11)
    left_elbow_angle = round(left_elbow_angle, 2)
    right_elbow_angle = calculateAngle(point_8, point_10, point_12)
    right_elbow_angle = round(right_elbow_angle, 2)

    def getWeight():

        filename = "data.txt"  # Replace with the actual file name

        data = {}  # Dictionary to store the extracted values

        # Open the file in read mode
        with open(filename, "r") as file:
            # Read each line in the file
            for line in file:
                # Split the line into key and value using the ":" delimiter
                key, value = line.strip().split(":")
                # Remove leading/trailing whitespaces from the key and value
                key = key.strip()
                value = value.strip()
                # Store the key-value pair in the data dictionary
                data[key] = value

        # Extract the values to variables
        weight = int(data.get("weight"))

        return weight

    def calculate_bmi(weight_kg, height_m):
        bmi = weight_kg / (height_m ** 2)
        return bmi

    # Example usage:
    weight = getWeight()  # Replace with the person's weight in kilograms
    height = (dist10 / 100)  # Replace with the person's height in meters
    height = round(height, 2)
    bmi = calculate_bmi(weight, height)

    distances = f"BMI: {bmi:.2f} kg/m2" + "\n" + f"Between Shoulder: {dist1} cm" + "\n" + f"Left Upper Arm: {dist2} cm" + "\n" + f"Right Upper Arm: {dist3} cm" + "\n" + f"Left Forearm : {dist4} cm" + "\n" + f"Right Forearm: {dist5} cm" + "\n" + f"Left Thigh: {dist6} cm" + "\n" + f"Right Thigh: {dist7} cm" + "\n" + f"Left Shin: {dist8} cm" + "\n" + f"Right Shin: {dist9} cm" + "\n" + f"Upper body segment: {dist11} cm" + "\n" + f"Lower body segment: {dist12} cm" + "\n" + f"Upper to lower segment ratio: {ratio12}" + "\n" + f"Full Tall: {dist10} cm" + "\n" + f"Left elbow angle: {left_elbow_angle} " + "\n" + f"Right elbow angle: {right_elbow_angle}"

    print(distances)
    #
    # # Split the 'distances' string into lines
    # distances_list = distances.split('\n')
    #
    # # Initialize an empty HTML table
    # table_html = "<table>\n"
    #
    # table_html += """<tr><th style="padding:10px;"colspan="2">Front distances and Angles</th></tr>"""
    # # Generate the HTML table row by row
    # for line in distances_list:
    #     part, value = line.split(': ')
    #     table_html += "<tr>"
    #     table_html += f"<td>{part}</td>"
    #     table_html += f"<td>{value}</td>"
    #     table_html += "</tr>\n"
    #
    # # Close the table tag
    # table_html += "</table>"
    #
    # # Print the HTML table or use it as needed
    # print(table_html)

    with open("Frontdistances.txt", mode='w') as f:  # I add the mode='w'
        f.write("Front distances and Angles\n\n"+distances)

    with open("Frontpoints.txt", mode='w') as f:  # I add the mode='w'
        for i in range(len(pointsall)):
            f.write("%f    " % float(pointsall[i][0].item()))
            f.write("%f    " % float(pointsall[i][1].item()))
            f.write("%f    \n" % float(pointsall[i][2].item()))

    data1 = np.loadtxt("Frontpoints.txt")
    pcdext = o3d.io.read_point_cloud("Frontpoints.txt", format="xyz")
    o3d.io.write_point_cloud("Frontpoints.ply", pcdext)
    mesh = o3d.io.read_point_cloud("Frontpoints.ply")
    array = np.asarray(mesh.points)

    def create_geometry_at_points(pointsfinal):
        geometries = o3d.geometry.TriangleMesh()
        for array in pointsfinal:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)  # create a small sphere to represent point
            sphere.translate(array)  # translate this sphere to point
            geometries += sphere
        geometries.paint_uniform_color([1.0, 0.0, 1.0])
        return geometries

    highlight_pnts = create_geometry_at_points(mesh.points)

    o3d.visualization.draw_geometries([body, highlight_pnts], window_name='Front Result')
