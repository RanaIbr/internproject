import open3d as o3d
import numpy as np
import copy
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

body = o3d.io.read_point_cloud("cropped_1.ply")
mesh_array = np.asarray(body.points)
colors = np.asarray(body.colors)
o3d.visualization.draw_geometries([body],window_name="Full body", width = 800, height = 800, left=800, top=200)

original_rotation_angles = (np.radians(90), 0, np.radians(-270))
R_original = body.get_rotation_matrix_from_xyz(original_rotation_angles)
R_inv = np.linalg.inv(R_original)
body.rotate(R_inv, center=(0, 0, 0))

#vis = o3d.visualization.VisualizerWithEditing()
#vis.create_window(window_name="Original Orientation",width = 800, height = 800, left=800, top=200)
#vis.add_geometry(body)
#vis.run()  # user picks points
#vis.capture_screen_image("file1.png", do_render=True)
#vis.destroy_window()


mesh_r=body
#normal estimation
mesh_r.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))

#o3d.visualization.draw_geometries([mesh_r],window_name="Full body", width = 800, height = 800, left=800, top=200)

array2=np.asarray(mesh_r.points)
print(array2)
colors=np.asarray(mesh_r.colors)

#Crop body without head
#points=array2[0:10000]
#colors1=colors[0:10000]
points = array2[array2[:, 2] > 1]  # Adjust 'threshold' to select points above a certain Z-coordinate
colors1 = colors[array2[:, 2] > 1]  # Adjust the range of colors accordingly


with open("body.txt", mode='w') as f:  
    for i in range(len(points)):
            f.write("%f    "%float(points[i][0].item()))
            f.write("%f    "%float(points[i][1].item()))
            f.write("%f    \n"%float(points[i][2].item()))

with open("colors1.txt", mode='w') as f:  
    for i in range(len(colors1)):
            f.write("%f    "%float(colors1[i][0].item()))
            f.write("%f    "%float(colors1[i][1].item()))
            f.write("%f    \n"%float(colors1[i][2].item()))

           
data = np.loadtxt("body.txt")
colors1 = np.loadtxt("colors1.txt")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data)
pcd.colors = o3d.utility.Vector3dVector(colors1)

o3d.io.write_point_cloud("3D_body.ply", pcd)
pcd_2 = o3d.io.read_point_cloud("3D_body.ply")

# Rotate to the Front side
mesh_r = copy.deepcopy(pcd_2)
R_front = pcd_2.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
mesh_r.rotate(R_front, center=(0, 0, 0))

vis = o3d.visualization.VisualizerWithEditing()
vis.create_window(window_name="body human",width = 800, height = 800, left=800, top=200)
vis.add_geometry(mesh_r)
vis.run()  # user picks points
vis.capture_screen_image("image-body.jpg", do_render=True)
vis.destroy_window()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

def detectPose(image, pose, output_filename):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks_2d = []

    if results.pose_landmarks:
        selected_landmarks = results.pose_landmarks.landmark[11:34]

        for landmark in selected_landmarks:
            x, y, _ = int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)
            cv2.circle(output_image, (x, y), 5, (0, 255, 0), -1)
            landmarks_2d.append([x, y])
        

    # Display the image with landmarks
    cv2.imshow("Image with Landmarks", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()     

    return landmarks_2d


image = cv2.imread('image-body.jpg')
landmarks_2d = detectPose(image, pose, 'pose_landmarks.txt')

# Convert 2D landmarks to 3D points (assuming all points are on the same plane)
landmarks_3d = np.zeros((len(landmarks_2d), 3))
landmarks_3d[:, :2] = landmarks_2d

# Create a point cloud from the 3D landmarks
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(landmarks_3d)

# Visualize the point cloud
#o3d.visualization.draw_geometries([point_cloud],window_name="body without head",width = 800, height = 800, left=800, top=200)

def create_geometry_at_points(point_cloud):
    geometries = o3d.geometry.TriangleMesh()
    for array in point_cloud:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5) #create a small sphere to represent point
        sphere.translate(array) #translate this sphere to point
        geometries += sphere
    geometries.paint_uniform_color([0, 255, 0])
    return geometries

highlight_pnts = create_geometry_at_points(point_cloud.points)


o3d.visualization.draw_geometries([highlight_pnts],window_name="Body Landmarks",
                                  width = 800, height = 800, left=800, top=200)


