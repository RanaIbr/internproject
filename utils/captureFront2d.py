import open3d as o3d
import numpy as np
import copy
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils


def detectPose(image, pose, output_filename, display=True):
    output_image = image.copy()

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(imageRGB)

    height, width, _ = image.shape

    landmarks = []

    if results.pose_landmarks:
        # print(results.pose_landmarks)
        # mp_drawing.draw_landmarks(image=output_image,
        #                           landmark_list=results.pose_landmarks,
        #                           connections=mp_pose.POSE_CONNECTIONS)
        # # Filter landmarks from start_landmark to end_landmark
        selected_landmarks = results.pose_landmarks.landmark[11:33 + 1]

        for landmark in selected_landmarks:
            x, y, z = int(landmark.x * width), int(landmark.y * height), landmark.z * width
            cv2.circle(output_image, (x, y), 5, (0, 255, 0), -1)

    if display:
        plt.title("Output Image")
        plt.axis('off')
        plt.show()  # Display the plot

    # Save the image with landmarks
    cv2.imwrite(output_filename, output_image)


def start_front(file_path):
    body = o3d.io.read_point_cloud(file_path)

    mesh_r = copy.deepcopy(body)
    R = body.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
    mesh_r.rotate(R, center=(0, 0, 0))

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=800, height=800, left=800, top=200)
    vis.add_geometry(mesh_r)
    vis.run()  # user picks points
    vis.capture_screen_image("outputs/file1.png", do_render=True)
    vis.destroy_window()

    image = cv2.imread('outputs/file1.png')
    output_filename = 'outputs/output_image_front.png'
    detectPose(image, pose, output_filename, display=False)

    image = cv2.imread('outputs/output_image_front.png')
    plt.figure(figsize=[7,7])
    plt.title("Output Image Front")
    plt.axis('off')
    imgplot = plt.imshow(image)
    plt.show()
    
