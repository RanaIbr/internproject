import open3d as o3d
import copy
import numpy as np
import math

body = o3d.io.read_point_cloud("cropped_1.ply")

vis = o3d.visualization.VisualizerWithEditing()
vis.create_window(window_name="Full Body clean",
                  width=800, height=800, left=800, top=200)
vis.add_geometry(body)
vis.run()
vis.destroy_window()
vis.get_picked_points()

# Get the indices of the picked points
picked_indices = vis.get_picked_points()

pointcloud_as_array = np.asarray(body.points)
points = pointcloud_as_array[vis.get_picked_points()]
pointsall = points[:, :]
print(f"Pointall: {pointsall}")

# Define the bounding box around the landmarks
min_bound = np.min(pointsall, axis=0)
max_bound = np.max(pointsall, axis=0)

# Create an AxisAlignedBoundingBox
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

# Crop the face region from the original mesh using the bounding box
part_region = body.crop(bbox)
# Save the cropped region as a .ply file
o3d.io.write_point_cloud("part_region.ply", part_region)
part_region1 = o3d.io.read_point_cloud("part_region.ply")

# o3d.visualization.draw_geometries([part_region1],window_name="part_region",
#                                 width = 800, height = 800, left=800, top=200)
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window(window_name="part_region",
                  width=800, height=800, left=800, top=200)
vis.add_geometry(part_region1)
vis.run()
vis.destroy_window()
vis.get_picked_points()

pointcloud_as_array1 = np.asarray(part_region1.points)
points1 = pointcloud_as_array1[vis.get_picked_points()]
pointsall1 = points1[:, :]

# point_1 = pointsall[0, :]
# point_2 = pointsall[1, :]
print(f"Pointall: {pointsall}")


def calculate_bmi(weight_kg, height_m):
    return weight_kg / (height_m ** 2)


def is_obese(bmi, bmi_threshold=30):
    return bmi >= bmi_threshold


def has_abdominal_obesity(waist_circumference_cm, waist_threshold_male=102, waist_threshold_female=88):
    # For simplicity, assuming gender-neutral threshold for waist circumference
    return waist_circumference_cm >= waist_threshold_male


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


# Calculate the waist circumference
circumference = np.linalg.norm(np.roll(pointsall, 1, axis=0) - pointsall, axis=1).sum()

# Convert the circumference to centimeters (assuming the data is in millimeters)
waist_circumference_cm = circumference * 0.1
print(f"circumference_cm: {waist_circumference_cm} cm")

z_max = max(body.points, key=lambda x: x[1])
z_min = min(body.points, key=lambda x: x[1])
dist9 = (euclidean_distance(z_max, z_min)) * 0.1
dist9 = round(dist9, 2)
print(f"Full Tall: {dist9} cm")

# Example data (replace with actual data)
weight = input("Please enter your weight in Kg: ")  # Replace with the person's weight in kilograms
weight_kg = float(weight)
height_m = (dist9 / 100)
height_m = round(height_m, 2)
# waist_circumference_cm = 90

# Calculate BMI
bmi = calculate_bmi(weight_kg, height_m)
print(f"BMI: {bmi:.2f} kg/m2")

# Define thresholds
bmi_threshold = 30
waist_threshold_male = 102
waist_threshold_female = 88

# Predict obesity and abdominal obesity
obese = is_obese(bmi, bmi_threshold)
abdominal_obesity = has_abdominal_obesity(waist_circumference_cm, waist_threshold_male)

if obese:
    print("The person is obese.")
else:
    print("The person is not obese.")

if abdominal_obesity:
    print("The person has abdominal obesity.")
else:
    print("The person does not have abdominal obesity.")

with open("points.txt", mode='w') as f:  # I add the mode='w'
    for i in range(len(pointsall)):
        f.write("%f    " % float(pointsall[i][0].item()))
        f.write("%f    " % float(pointsall[i][1].item()))
        f.write("%f    \n" % float(pointsall[i][2].item()))

data1 = np.loadtxt("points.txt")
pcdext = o3d.io.read_point_cloud("points.txt", format="xyz")
o3d.io.write_point_cloud("points.ply", pcdext)
mesh = o3d.io.read_point_cloud("points.ply")
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
o3d.visualization.draw_geometries([part_region, highlight_pnts], window_name="Full body",
                                  width=800, height=800, left=800, top=200)