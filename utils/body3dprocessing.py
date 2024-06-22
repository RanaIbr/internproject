import open3d as o3d
import numpy as np
import copy
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import math

# Read the 3D point cloud data from a PLY file
body = o3d.io.read_point_cloud("cropped_1.ply")

# Front body display
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window(window_name="Front side", width=800, height=800, left=800, top=200)
vis.add_geometry(body)
vis.run()  # User picks points
vis.capture_screen_image("image-body-front.jpg", do_render=True)
vis.destroy_window()

# Back body display
mesh_b = copy.deepcopy(body)

# Create rotation matrix for 180 degree rotation around Y-axis
R = np.eye(3)
R[0, 0] = -1
R[2, 2] = -1

mesh_b.rotate(R, center=mesh_b.get_center())  # Rotate 180 degrees around Y-axis for back side

vis = o3d.visualization.VisualizerWithEditing()
vis.create_window(window_name="Back side", width=800, height=800, left=800, top=200)
vis.add_geometry(mesh_b)
vis.run()  # User picks points
vis.capture_screen_image("image-body-back.jpg", do_render=True)
vis.destroy_window()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Read the front and back images
image_front = cv2.imread('image-body-front.jpg')
image_back = cv2.imread('image-body-back.jpg')

image_rows, image_cols, _ = image_front.shape
# Define the selected indices and their corresponding names
selected_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
selected_landmark_names = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                           "left_wrist", "right_wrist", "left_hip", "right_hip",
                           "left_knee", "right_knee", "left_ankle", "right_ankle",
                           "left_foot_index", "right_foot_index"]

# Process the image
results = pose.process(cv2.cvtColor(image_front, cv2.COLOR_BGR2RGB))

# Initialize a list to store 2D landmarks
landmarks_2d = []


# Function to detect and extract 2D landmarks
def detectSelectedPoseWithNames(image, pose, selected_indices, landmark_names):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    print(results)
    landmarks_2d = []
    if results.pose_landmarks:
        for i, index in enumerate(selected_indices):
            landmark_name = selected_landmark_names[i]
            landmark = results.pose_landmarks.landmark[index]
            x, y, _ = int(landmark.x * image_cols), int(landmark.y * image_rows), int(landmark.z * image_cols)
            print(f"Landmark: {landmark_name} - X: {x}, Y: {y}")
            cv2.circle(output_image, (x, y), 5, (0, 255, 0), -1)  # Adjust the circle radius here
            cv2.putText(output_image, landmark_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            landmarks_2d.append([x, y])

    # Display the image with selected landmarks and names
    cv2.imshow("Image with Selected Landmarks", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return landmarks_2d


# Detect and extract 2D landmarks from front and back images
landmarks_front = detectSelectedPoseWithNames(image_front, pose, selected_indices, selected_landmark_names)
landmarks_back = detectSelectedPoseWithNames(image_back, pose, selected_indices, selected_landmark_names)

# Define a scaling factor
scale_factor = 3  # Adjust this value to increase or decrease the size of the landmarks

# Scale the 2D landmarks
landmarks_2d_scaled = [[int(x * scale_factor), int(y * scale_factor)] for x, y in landmarks_front]

# Create a point cloud from the scaled 2D landmarks
landmarks_3d_scaled = np.zeros((len(landmarks_2d_scaled), 3))
landmarks_3d_scaled[:, :2] = landmarks_2d_scaled
print(landmarks_3d_scaled)
# Create a point cloud from the scaled 3D landmarks
point_cloud_scaled = o3d.geometry.PointCloud()
point_cloud_scaled.points = o3d.utility.Vector3dVector(landmarks_3d_scaled)
o3d.io.write_point_cloud("pose_landmarks_scaled.ply", point_cloud_scaled)

landmarks = o3d.io.read_point_cloud("pose_landmarks_scaled.ply")
R = landmarks.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
landmarks.rotate(R, center=(0, 0, 0))


def create_geometry_at_points(array):
    geometries = o3d.geometry.TriangleMesh()
    for array in array:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=15)  # create a small sphere to represent point
        sphere.translate(array)  # translate this sphere to point
        geometries += sphere
    geometries.paint_uniform_color([0, 1, 0])
    return geometries


highlight_pnts1 = create_geometry_at_points(landmarks.points)
o3d.visualization.draw_geometries([highlight_pnts1], window_name="Landmarks body",
                                  width=800, height=800, left=800, top=200)


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


# Calculate the Euclidean distance between left shoulder and right shoulder
left_shoulder = landmarks_3d_scaled[selected_landmark_names.index("left_shoulder")]
right_shoulder = landmarks_3d_scaled[selected_landmark_names.index("right_shoulder")]

left_elbow = landmarks_3d_scaled[selected_landmark_names.index("left_elbow")]
right_elbow = landmarks_3d_scaled[selected_landmark_names.index("right_elbow")]

left_wrist = landmarks_3d_scaled[selected_landmark_names.index("left_wrist")]
right_wrist = landmarks_3d_scaled[selected_landmark_names.index("right_wrist")]

left_hip = landmarks_3d_scaled[selected_landmark_names.index("left_hip")]
left_knee = landmarks_3d_scaled[selected_landmark_names.index("left_knee")]

right_hip = landmarks_3d_scaled[selected_landmark_names.index("right_hip")]
right_knee = landmarks_3d_scaled[selected_landmark_names.index("right_knee")]

left_ankle = landmarks_3d_scaled[selected_landmark_names.index("left_ankle")]
right_ankle = landmarks_3d_scaled[selected_landmark_names.index("right_ankle")]

z_max = max(body.points, key=lambda x: x[1])
z_min = min(body.points, key=lambda x: x[1])

dist1 = (euclidean_distance(left_shoulder, left_elbow)) * 0.1
dist1 = round(dist1, 2)
print(f"Left Upper Arm: {dist1} cm")

dist2 = (euclidean_distance(right_shoulder, right_elbow)) * 0.1
dist2 = round(dist2, 2)
print(f"Right Upper Arm: {dist2} cm")

dist3 = (euclidean_distance(left_elbow, left_wrist)) * 0.1
dist3 = round(dist3, 2)
print(f"Left Forearm: {dist3} cm")

dist4 = (euclidean_distance(right_elbow, right_wrist)) * 0.1
dist4 = round(dist4, 2)
print(f"Right Forearm: {dist4} cm")

dist5 = (euclidean_distance(left_hip, left_knee)) * 0.1
dist5 = round(dist5, 2)
print(f"Left Thigh: {dist5} cm")

dist6 = (euclidean_distance(right_hip, right_knee)) * 0.1
dist6 = round(dist6, 2)
print(f"Right Thigh: {dist6} cm")

dist7 = (euclidean_distance(left_knee, left_ankle)) * 0.1
dist7 = round(dist7, 2)
print(f"Left Shin: {dist7} cm")

dist8 = (euclidean_distance(right_knee, right_ankle)) * 0.1
dist8 = round(dist8, 2)
print(f"Right Shin: {dist8} cm")

dist9 = (euclidean_distance(z_max, z_min)) * 0.1
dist9 = round(dist9, 2)
print(f"Full Tall: {dist9} cm")


def calculate_bmi(weight_kg, height_m):
    bmi = weight_kg / (height_m ** 2)
    return bmi


# Example usage:
weight = input("Please enter your weight in Kg: ")  # Replace with the person's weight in kilograms
weight = float(weight)
height = (dist9 / 100)  # Replace with the person's height in meters
height = round(height, 2)
bmi = calculate_bmi(weight, height)

print(f"BMI: {bmi:.2f} kg/m2")


def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360
    return angle

# left_elbow_angle = calculateAngle(left_hip, left_knee, left_ankle)
# left_elbow_angle = round(left_elbow_angle, 2)
# right_elbow_angle = calculateAngle(right_hip, right_knee, right_ankle)
# right_elbow_angle = round(right_elbow_angle, 2)

# print(f"Left elbow angle is: {left_elbow_angle} ")
# print(f"Right elbow angle is: {right_elbow_angle}")