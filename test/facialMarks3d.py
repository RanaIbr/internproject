import open3d as o3d
import open3d
import numpy as np
import copy
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt



# Read the point cloud
pcd = o3d.io.read_point_cloud("../resources/3d_models/Individual1.ply")
mesh_array = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

#Visualization of point cloud with window size
#vis = o3d.visualization.VisualizerWithEditing()
#vis.create_window(window_name="Full body",width = 800, height = 800, left=800, top=200)
#vis.add_geometry(pcd)
#vis.run()  
#vis.destroy_window()


points = mesh_array[mesh_array[:, 2] < 700]
colors1 = colors[mesh_array[:, 2] < 700]

with open("head.txt", mode='w') as f:  
    for i in range(len(points)):
            f.write("%f    "%float(points[i][0].item()))
            f.write("%f    "%float(points[i][1].item()))
            f.write("%f    \n"%float(points[i][2].item()))

with open("colors1.txt", mode='w') as f:  
    for i in range(len(colors1)):
            f.write("%f    "%float(colors1[i][0].item()))
            f.write("%f    "%float(colors1[i][1].item()))
            f.write("%f    \n"%float(colors1[i][2].item()))

           
data = np.loadtxt("head.txt")
colors1 = np.loadtxt("colors1.txt")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data)
pcd.colors = o3d.utility.Vector3dVector(colors1)

o3d.io.write_point_cloud("3D_Head.ply", pcd)
pcd_2 = o3d.io.read_point_cloud("3D_Head.ply")

# Rotate to the Front side
mesh_r = copy.deepcopy(pcd_2)
R_front = pcd_2.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
mesh_r.rotate(R_front, center=(0, 0, 0))

mesh_r.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
    #print(dir(head))
    
#o3d.visualization.draw_geometries([mesh_r],window_name="head",
 #                                         width = 800, height = 800, left=800, top=200)
    

#remove the outlier by index first time

voxel_size=0.05
pcd_downsampled=mesh_r.voxel_down_sample(voxel_size=voxel_size)
uni_down_pcd = pcd_downsampled.uniform_down_sample(every_k_points=3)

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    #outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="3D-Head",width = 800, height = 800, left=50, top=50)
    vis.add_geometry(inlier_cloud)
    vis.run()  # user picks points
    vis.capture_screen_image("image-head.jpg", do_render=True)
    vis.destroy_window()
    #o3d.visualization.draw_geometries([inlier_cloud],width=800,height=800,left=50,top=50)

print("Statistical oulier removal")
cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.5)
display_inlier_outlier(uni_down_pcd, ind)
o3d.io.write_point_cloud("3D-Head clean.ply", cl)



# Landmarks 3D face mesh extraction 
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

file = open('landmarks.txt', 'w')

# Load the static image
image_path = 'image-head.jpg'  # Change this to the path of your image
frame = cv2.imread(image_path)

image_rows, image_cols = frame.shape[:2]

# Initialize the face mesh model
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Process the image
results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if not results.multi_face_landmarks:
    print("No face detected in the image.")
else:
    annotated_image = frame.copy()
    face_landmarks = results.multi_face_landmarks[0]
    for landmark in face_landmarks.landmark:
        landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px:
            cv2.circle(annotated_image, landmark_px, 1, (255, 0, 0), -1)
            file.write(f'{landmark_px[0]}\t{landmark_px[1]}\t{(0.5 + landmark.z) * 550}')
            file.write('\n')

# Close the file
file.close()

# Save or display the annotated image
cv2.imwrite('annotated_image.jpg', annotated_image)
cv2.imshow('Annotated Image', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 3D visualization
pcd = open3d.io.read_point_cloud('landmarks.txt', format='xyz')

R = pcd.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
pcd.rotate(R, center=(0, 0, 0))

pcd.normals = open3d.utility.Vector3dVector(np.zeros((1, 3)))
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(10)

# pcd.colors = open3d.utility.Vector3dVector(colors)
with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
    rec_mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
rec_mesh.paint_uniform_color([0.3, 0.3, 0.3])
rec_mesh.compute_vertex_normals()
open3d.visualization.draw_geometries([pcd, rec_mesh], mesh_show_back_face=True,window_name="3D-Head",width = 800, height = 800, left=50, top=50)

pcd = rec_mesh.sample_points_poisson_disk(6000)
pcd.normals = open3d.utility.Vector3dVector(np.zeros((1, 3)))
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(10)
#open3d.visualization.draw_geometries([pcd], point_show_normal=True)

with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
    rec_mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
rec_mesh.compute_vertex_normals()
#open3d.visualization.draw_geometries([rec_mesh], mesh_show_back_face=True,window_name="3D-Head",width = 800, height = 800, left=50, top=50)


data1 = np.loadtxt("landmarks.txt")
pcdext = o3d.io.read_point_cloud("landmarks.txt", format="xyz")
o3d.io.write_point_cloud("landmarks.ply", pcdext)
mesh = o3d.io.read_point_cloud("landmarks.ply")
array=np.asarray(mesh.points)



def create_geometry_at_points(array):
    geometries = o3d.geometry.TriangleMesh()
    for array in array:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1) #create a small sphere to represent point
        sphere.translate(array) #translate this sphere to point
        geometries += sphere
    geometries.paint_uniform_color([0.0, 0.0, 0.0])
    return geometries

highlight_pnts = create_geometry_at_points(mesh.points)

o3d.visualization.draw_geometries([highlight_pnts],window_name="3D Face Mesh",
                                  width = 800, height = 800, left=800, top=200)



