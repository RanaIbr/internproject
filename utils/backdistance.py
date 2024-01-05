import numpy as np
import open3d as o3d
import copy
from scipy.spatial import distance
import math


def backLandmarks(file_path):
    body = o3d.io.read_point_cloud(file_path)

    # back body display
    mesh_b = copy.deepcopy(body)
    R2 = body.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-90)))
    mesh_b.rotate(R2, center=(0, 0, 0))

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

    point_1 = pointsall1[0, :]
    point_2 = pointsall1[1, :]
    point_3 = pointsall1[2, :]
    point_4 = pointsall1[3, :]
    point_5 = pointsall1[4, :]
    point_6 = pointsall1[5, :]
    point_7 = pointsall1[6, :]
    point_8 = pointsall1[7, :]
    point_9 = pointsall1[8, :]
    point_10 = pointsall1[9, :]
    point_11 = pointsall1[10, :]
    point_12 = pointsall1[11, :]

    point_13 = pointsall1[12, :]
    point_14 = pointsall1[13, :]
    point_15 = pointsall1[14, :]
    point_16 = pointsall1[15, :]
    point_17 = pointsall1[16, :]
    point_18 = pointsall1[17, :]
    point_19 = pointsall1[18, :]
    point_20 = pointsall1[19, :]
    point_21 = pointsall1[20, :]
    point_22 = pointsall1[21, :]

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

    dist11 = (distance.euclidean(point_22, z_min)) * 0.1
    dist11 = round(dist11, 2)

    dist12 = (distance.euclidean(point_22, z_max)) * 0.1
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

    back_neck_angle = calculateAngle(point_13, point_18, point_22)
    back_neck_angle = round(back_neck_angle, 2)

    distances = f"Between Shoulder Back: {dist1} cm" + "\n" + f"Left Upper Arm Back: {dist2} cm" + "\n" + f"Right Upper Arm Back: {dist3} cm" + "\n" + f"Left Forearm Back: {dist4} cm" + "\n" + f"Right Forearm Back: {dist5} cm" + "\n" + f"Left Thigh Back: {dist6} cm" + "\n" + f"Right Thigh Back: {dist7} cm" + "\n" + f"Left Shin Back: {dist8} cm" + "\n" + f"Right Shin Back: {dist9} cm" + "\n" + f"Upper body segment: {dist11} cm" + "\n" + f"Lower body segment: {dist12} cm" + "\n" + f"Upper to lower segment ratio: {ratio12}" + "\n" + f"Full Tall: {dist10} cm" + "\n" + f"Left elbow angle Back is: {left_elbow_angle} " + "\n" + f"Right elbow angle Back: {right_elbow_angle}" + "\n" + f"Back Neck Angle: {back_neck_angle}"

    # # Split the 'distances' string into lines
    # distances_list = distances.split('\n')
    #
    # # Initialize an empty HTML table
    # table_html = "<table>\n"
    # table_html += """<tr><th style="padding:10px;"colspan="2">Back distances and Angles</th></tr>"""
    #
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


    print(distances)

    with open("Backdistances.txt", mode='w') as f:  # I add the mode='w'
        f.write("\nBack distances and Angles\n\n"+distances)

    with open("Backpoints.txt", mode='w') as f:  # I add the mode='w'
        for i in range(len(pointsall1)):
            f.write("%f    " % float(pointsall1[i][0].item()))
            f.write("%f    " % float(pointsall1[i][1].item()))
            f.write("%f    \n" % float(pointsall1[i][2].item()))

    data2 = np.loadtxt("Backpoints.txt")
    pcdext1 = o3d.io.read_point_cloud("Backpoints.txt", format="xyz")
    o3d.io.write_point_cloud("Backpoints.ply", pcdext1)
    mesh1 = o3d.io.read_point_cloud("Backpoints.ply")
    array1 = np.asarray(mesh1.points)

    def create_geometry_at_points(pointsfinal1):
        geometries = o3d.geometry.TriangleMesh()
        for array1 in pointsfinal1:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)  # create a small sphere to represent point
            sphere.translate(array1)  # translate this sphere to point
            geometries += sphere
        geometries.paint_uniform_color([1.0, 0.0, 1.0])
        return geometries

    highlight_pnts1 = create_geometry_at_points(mesh1.points)

    o3d.visualization.draw_geometries([body, highlight_pnts1], window_name='Front Result')
