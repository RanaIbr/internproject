import numpy as np
import open3d as o3d
import copy
import math


def allDistancesFunction():
    body = o3d.io.read_point_cloud("resources/3d_models/cleanBody.ply")

    # back body display
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

    # --------------------------------------------------------------------

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

    # ----------------------------------------------------------------

    # Distances and angles of Front body

    def euclidean_distance(point1, point2):
        """
        Calculate the Euclidean distance between two points in 3D space.

        Parameters:
        - point1: Tuple or list representing the coordinates of the first point (x, y, z).
        - point2: Tuple or list representing the coordinates of the second point (x, y, z).

        Returns:
        - The Euclidean distance between the two points.
        """
        distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)
        return distance

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

    dist1 = (euclidean_distance(point_1, point_2)) * 0.1
    dist1 = round(dist1, 2)
    dist2 = (euclidean_distance(point_1, point_3)) * 0.1
    dist2 = round(dist2, 2)
    dist3 = (euclidean_distance(point_2, point_4)) * 0.1
    dist3 = round(dist3, 2)
    dist4 = (euclidean_distance(point_3, point_5)) * 0.1
    dist4 = round(dist4, 2)
    dist5 = (euclidean_distance(point_4, point_6)) * 0.1
    dist5 = round(dist5, 2)
    dist6 = (euclidean_distance(point_7, point_9)) * 0.1
    dist6 = round(dist6, 2)
    dist7 = (euclidean_distance(point_8, point_10)) * 0.1
    dist7 = round(dist7, 2)
    dist8 = (euclidean_distance(point_9, point_11)) * 0.1
    dist8 = round(dist8, 2)
    dist9 = (euclidean_distance(point_10, point_12)) * 0.1
    dist9 = round(dist9, 2)
    dist10 = (euclidean_distance(z_max, z_min)) * 0.1
    dist10 = round(dist10, 2)
    dist11 = (euclidean_distance(point_13, z_min)) * 0.1
    dist11 = round(dist11, 2)
    dist12 = (euclidean_distance(point_13, z_max)) * 0.1
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

    def calculate_bmi(weight_kg, height_m):
        bmi = weight_kg / (height_m ** 2)
        return bmi

    # Example usage:
    weight = 85  # Replace with the person's weight in kilograms
    height = (dist10 / 100)  # Replace with the person's height in meters
    height = round(height, 2)
    bmi = calculate_bmi(weight, height)

    print(f"Distances and angles of Front body:")
    print(f"BMI: {bmi:.2f} kg/m2")
    print(f"Between Shoulder: {dist1} cm")
    print(f"Left Upper Arm: {dist2} cm")
    print(f"Right Upper Arm: {dist3} cm")
    print(f"Left Forearm : {dist4} cm")
    print(f"Right Forearm: {dist5} cm")
    print(f"Left Thigh: {dist6} cm")
    print(f"Right Thigh: {dist7} cm")
    print(f"Left Shin: {dist8} cm")
    print(f"Right Shin: {dist9} cm")
    print(f"Upper body segment: {dist11} cm")
    print(f"Lower body segment: {dist12} cm")
    print(f"Upper to lower segment ratio: {ratio12}")
    print(f"Left elbow angle is: {left_elbow_angle} ")
    print(f"Right elbow angle is: {right_elbow_angle}")
    print(f"Full Tall: {dist10} cm")
    print(f" \n")

    distances = f"Distances and angles of Front body:\nBMI: {bmi:.2f} kg/m2" + "\n" + f"Between Shoulder: {dist1} cm" + "\n" + f"Left Upper Arm: {dist2} cm" + "\n" + f"Right Upper Arm: {dist3} cm" + "\n" + f"Left Forearm : {dist4} cm" + "\n" + f"Right Forearm: {dist5} cm" + "\n" + f"Left Thigh: {dist6} cm" + "\n" + f"Right Thigh: {dist7} cm" + "\n" + f"Left Shin: {dist8} cm" + "\n" + f"Right Shin: {dist9} cm" + "\n" + f"Upper body segment: {dist11} cm" + "\n" + f"Lower body segment: {dist12} cm" + "\n" + f"Upper to lower segment ratio: {ratio12}" + "\n" + f"Full Tall: {dist10} cm" + "\n" + f"Left elbow angle: {left_elbow_angle} " + "\n" + f"Right elbow angle: {right_elbow_angle}"
    distances_html = f"""
    
    <table>
  <tr>
    <td colspan=2>Distances and angles of Front body:</td>
  </tr>
  <tr>
    <td>bmi:</td>
    <td>{bmi:.2f} kg/m2</td>
  </tr>
  <tr>
    <td>Between Shoulder:</td>
    <td>{dist1} cm</td>
  </tr>
  <tr>
    <td>Left Upper Arm:</td>
    <td>{dist2} cm</td>
  </tr>
  <tr>
    <td>Right Upper Arm:</td>
    <td>{dist3} cm</td>
  </tr>
  <tr>
    <td>Left Forearm:</td>
    <td>{dist4} cm</td>
  </tr>
  <tr>
    <td>Right Forearm:</td>
    <td>{dist5} cm</td>
  </tr>
  <tr>
    <td>Left Thigh:</td>
    <td>{dist6} cm</td>
  </tr>
  <tr>
    <td>Right Thigh:</td>
    <td>{dist7} cm</td>
  </tr>
  <tr>
    <td>Left Shin:</td>
    <td>{dist8} cm</td>
  </tr>
  <tr>
    <td>Right Shin:</td>
    <td>{dist9} cm</td>
  </tr>
  <tr>
    <td>Upper body segment:</td>
    <td>{dist11} cm</td>
  </tr>
  <tr>
    <td>Lower body segment:</td>
    <td>{dist12} cm</td>
  </tr>
  <tr>
    <td>Upper to lower segment ratio:</td>
    <td>{ratio12}</td>
  </tr>
  <tr>
    <td>Left elbow angle:</td>
    <td>{left_elbow_angle}</td>
  </tr>
  <tr>
    <td>Right elbow angle:</td>
    <td>{right_elbow_angle}</td>
  </tr>
  <tr>
    <td>Full Tall:</td>
    <td>{dist10} cm</td>
  </tr>
</table>



    """
    with open("outputs/text_files/Frontdistances.txt", mode='w') as f:  # I add the mode='w'
        f.write("Front distances and Angles\n\n" + distances)

    with open("outputs/text_files/FrontdistancesHtml.txt", mode='w') as f:  # I add the mode='w'
        f.write(distances_html)
    # Distances and angles of Back body

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

    dist1 = (euclidean_distance(point_1, point_2)) * 0.1
    dist1 = round(dist1, 2)
    dist2 = (euclidean_distance(point_1, point_3)) * 0.1
    dist2 = round(dist2, 2)
    dist3 = (euclidean_distance(point_2, point_4)) * 0.1
    dist3 = round(dist3, 2)
    dist4 = (euclidean_distance(point_3, point_5)) * 0.1
    dist4 = round(dist4, 2)
    dist5 = (euclidean_distance(point_4, point_6)) * 0.1
    dist5 = round(dist5, 2)
    dist6 = (euclidean_distance(point_7, point_9)) * 0.1
    dist6 = round(dist6, 2)
    dist7 = (euclidean_distance(point_8, point_10)) * 0.1
    dist7 = round(dist7, 2)
    dist8 = (euclidean_distance(point_9, point_11)) * 0.1
    dist8 = round(dist8, 2)
    dist9 = (euclidean_distance(point_10, point_12)) * 0.1
    dist9 = round(dist9, 2)
    dist10 = (euclidean_distance(point_13, z_min)) * 0.1
    dist10 = round(dist10, 2)
    dist11 = (euclidean_distance(point_13, z_max)) * 0.1
    dist11 = round(dist11, 2)
    ratio21 = dist10 / dist11
    ratio21 = round(ratio12, 2)

    left_elbow_angle = calculateAngle(point_7, point_9, point_11)
    left_elbow_angle = round(left_elbow_angle, 2)
    right_elbow_angle = calculateAngle(point_8, point_10, point_12)
    right_elbow_angle = round(right_elbow_angle, 2)

    back_neck_angle = calculateAngle(point_13, point_14, point_15)
    back_neck_angle = round(back_neck_angle, 2)

    print(f"Distances and angles of Back body:")
    print(f"Between Shoulder Back: {dist1} cm")
    print(f"Left Upper Arm Back: {dist2} cm")
    print(f"Right Upper Arm Back: {dist3} cm")
    print(f"Left Forearm Back: {dist4} cm")
    print(f"Right Forearm Back: {dist5} cm")
    print(f"Left Thigh Back: {dist6} cm")
    print(f"Right Thigh Back: {dist7} cm")
    print(f"Left Shin Back: {dist8} cm")
    print(f"Right Shin Back: {dist9} cm")
    print(f"Upper body segment: {dist10} cm")
    print(f"Lower body segment: {dist11} cm")
    print(f"Upper to lower segment ratio: {ratio21}")
    print(f"Left elbow angle Back is: {left_elbow_angle} ")
    print(f"Right elbow angle Back is: {right_elbow_angle}")
    print(f"back_neck_angle is: {back_neck_angle}")

    distances = f"Between Shoulder Back: {dist1} cm" + "\n" + f"Left Upper Arm Back: {dist2} cm" + "\n" + f"Right Upper Arm Back: {dist3} cm" + "\n" + f"Left Forearm Back: {dist4} cm" + "\n" + f"Right Forearm Back: {dist5} cm" + "\n" + f"Left Thigh Back: {dist6} cm" + "\n" + f"Right Thigh Back: {dist7} cm" + "\n" + f"Left Shin Back: {dist8} cm" + "\n" + f"Right Shin Back: {dist9} cm" + "\n" + f"Upper body segment: {dist11} cm" + "\n" + f"Lower body segment: {dist12} cm" + "\n" + f"Upper to lower segment ratio: {ratio12}" + "\n" + "\n" + f"Left elbow angle Back is: {left_elbow_angle} " + "\n" + f"Right elbow angle Back: {right_elbow_angle}" + "\n" + f"Back Neck Angle: {back_neck_angle}"
    distances_html = f"""

    <table border=1>
   <tr>
    <td colspan=2>Distances and angles of Back body:</td>
  </tr>
  <tr>
    <td>Between Shoulder Back:</td>
    <td>{dist1} cm</td>
  </tr>
  <tr>
    <td>Left Upper Arm Back:</td>
    <td>{dist2} cm</td>
  </tr>
  <tr>
    <td>Right Upper Arm Back:</td>
    <td>{dist3} cm</td>
  </tr>
  <tr>
    <td>Left Forearm Back:</td>
    <td>{dist4} cm</td>
  </tr>
  <tr>
    <td>Right Forearm Back:</td>
    <td>{dist5} cm</td>
  </tr>
  <tr>
    <td>Left Thigh Back:</td>
    <td>{dist6} cm</td>
  </tr>
  <tr>
    <td>Right Thigh Back:</td>
    <td>{dist7} cm</td>
  </tr>
  <tr>
    <td>Left Shin Back:</td>
    <td>{dist8} cm</td>
  </tr>
  <tr>
    <td>Right Shin Back:</td>
    <td>{dist9} cm</td>
  </tr>
  <tr>
    <td>Upper body segment:</td>
    <td>{dist10} cm</td>
  </tr>
  <tr>
    <td>Lower body segment:</td>
    <td>{dist11} cm</td>
  </tr>
  <tr>
    <td>Upper to lower segment ratio:</td>
    <td>{ratio21}</td>
  </tr>
  <tr>
    <td>Left elbow angle Back is:</td>
    <td>{left_elbow_angle}</td>
  </tr>
  <tr>
    <td>Right elbow angle Back is:</td>
    <td>{right_elbow_angle}</td>
  </tr>
  <tr>
    <td>back_neck_angle is:</td>
    <td>{back_neck_angle}</td>
  </tr>
</table>

    """


    with open("outputs/text_files/Backdistances.txt", mode='w') as f:  # I add the mode='w'
        f.write("\nBack distances and Angles\n\n" + distances)

    with open("outputs/text_files/BackdistancesHtml.txt", mode='w') as f:  # I add the mode='w'
        f.write(distances_html)

    # ----------------------------------------------------------------

    def create_geometry_at_points(pointsfinal):
        geometries = o3d.geometry.TriangleMesh()
        for array in pointsfinal:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)  # create a small sphere to represent point
            sphere.translate(array)  # translate this sphere to point
            geometries += sphere
        geometries.paint_uniform_color([1.0, 0.0, 1.0])
        return geometries

    def create_geometry_at_points(pointsfinal1):
        geometries = o3d.geometry.TriangleMesh()
        for array1 in pointsfinal1:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)  # create a small sphere to represent point
            sphere.translate(array1)  # translate this sphere to point
            geometries += sphere
        geometries.paint_uniform_color([1.0, 0.0, 1.0])
        return geometries

    highlight_pnts = create_geometry_at_points(mesh.points)
    highlight_pnts1 = create_geometry_at_points(mesh1.points)

    o3d.visualization.draw_geometries([body, highlight_pnts, highlight_pnts1], window_name="Full body", width=800,
                                      height=800, left=800, top=200)
