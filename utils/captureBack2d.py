import open3d as o3d
import numpy as np
import copy
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Load the pose estimation model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils


# Function to detect pose and save the image
def detectPose(image, pose, output_filename, display=True):
    output_image = image.copy()

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the rotated image (back side)
    results = pose.process(imageRGB)

    height, width, _ = image.shape

    landmarks = []

    if results.pose_landmarks:

        selected_landmarks = results.pose_landmarks.landmark[11:33 + 1]

        for landmark in selected_landmarks:
            x, y, z = int(landmark.x * width), int(landmark.y * height), landmark.z * width
            cv2.circle(output_image, (x, y), 5, (0, 255, 0), -1)
    if display:
        # Display the image with landmarks
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')
        plt.show()  # Display the plot

    # Save the image with landmarks
    cv2.imwrite(output_filename, output_image)


def start_back(file_path):
    # Load the 3D body model
    body = o3d.io.read_point_cloud(file_path)

    # Rotate the 3D body to show the back side
    mesh_r = copy.deepcopy(body)
    R = body.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-90)))  # Adjust rotation as needed
    mesh_r.rotate(R, center=(0, 0, 0))

    # Create an Open3D visualizer
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=800, height=800, left=800, top=200)
    vis.add_geometry(mesh_r)

    # Set the camera view to the back side of the 3D body
    ctr = vis.get_view_control()
    ctr.rotate(180.0, 0.0)  # Rotate the view 180 degrees around the Y-axis (back side)

    # Run the visualizer for user interaction (e.g., picking points)
    vis.run()

    # Capture the screen image
    vis.capture_screen_image("outputs/file1.png", do_render=True)

    # Close the visualizer window
    vis.destroy_window()

    # Load the captured image and process it for pose estimation
    image = cv2.imread('outputs/file1.png')
    output_filename = 'outputs/output_image_back.png'
    detectPose(image, pose, output_filename, display=False)


    image = cv2.imread('outputs/output_image_back.png')
    plt.figure(figsize=[7,7])
    plt.title("Output Image back")
    plt.axis('off')
    imgplot = plt.imshow(image)
    plt.show()    
