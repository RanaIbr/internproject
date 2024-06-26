import open3d as o3d
import open3d
import numpy as np
import copy
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import open3d.visualization.gui as gui
import time


def facial():
    # Landmarks 3D face mesh extraction
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    file = open('../outputs/text_files/landmarks.txt', 'w')

    # Load the static image
    image_path = '../outputs/images/image-head.jpg'  # Change this to the path of your image
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
                cv2.circle(annotated_image, landmark_px, 1, (0, 255, 0), -1)
                file.write(f'{landmark_px[0]}\t{landmark_px[1]}\t{(0.5 + landmark.z) * 550}')
                file.write('\n')

    # Close the file
    file.close()

    # Save or display the annotated image
    # cv2.imwrite('annotated_image.jpg', annotated_image)
    # cv2.imshow('Annotated Image', annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    data1 = np.loadtxt("../outputs/text_files/landmarks.txt")
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(data1)
    o3d.io.write_point_cloud("../outputs/3d_models/landmarks.ply", pcd1)

    # Read point clouds
    body = o3d.io.read_point_cloud("../outputs/3d_models/3D_Head clean.ply")
    body1 = o3d.io.read_point_cloud("../outputs/3d_models/landmarks.ply")
    R = body1.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
    body1.rotate(R, center=(0, 0, 0))

    # Get max points
    z_max = max(body.points, key=lambda x: x[2])
    z_max1 = max(body1.points, key=lambda x: x[2])

    # Create sphere at centroid position
    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    centroid_sphere.translate(z_max)

    centroid_sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    centroid_sphere1.translate(z_max1)

    # Create axes LineSets
    def create_axes_line_set(origin):
        axes_line_set = o3d.geometry.LineSet()
        axes_points = np.array([origin, origin + [50, 0, 0], origin + [0, 50, 0], origin + [0, 0, 50]])
        axes_lines = [[0, 1], [0, 2], [0, 3]]
        axes_line_set.points = o3d.utility.Vector3dVector(axes_points)
        axes_line_set.lines = o3d.utility.Vector2iVector(axes_lines)
        axes_line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return axes_line_set

    axes_line_set = create_axes_line_set(z_max)
    axes_line_set1 = create_axes_line_set(z_max1)

    # Calculate translation and scale
    translation = z_max - z_max1
    scale_factor = 1.0  # Adjust as needed

    # Translate and scale landmarks
    body1.translate(translation)
    scaled_landmarks = o3d.geometry.PointCloud()
    scaled_landmarks.points = o3d.utility.Vector3dVector(np.asarray(body1.points) * scale_factor)
    mesh_array = np.asarray(scaled_landmarks.points)
    print(mesh_array)

    with open("../outputs/text_files/scaled_landmarks.txt", mode='w') as f:
        for i in range(len(mesh_array)):
            f.write("%f    " % float(mesh_array[i][0].item()))
            f.write("%f    " % float(mesh_array[i][1].item()))
            f.write("%f    \n" % float(mesh_array[i][2].item()))

    data2 = np.loadtxt("../outputs/text_files/scaled_landmarks.txt")
    pcdext = o3d.io.read_point_cloud("../outputs/text_files/scaled_landmarks.txt", format="xyz")
    o3d.io.write_point_cloud("../outputs/3d_models/scaled_landmarks.ply", pcdext)
    mesh2 = o3d.io.read_point_cloud("../outputs/3d_models/scaled_landmarks.ply")
    array = np.asarray(mesh2.points)

    def create_geometry_at_points(array):
        geometries = o3d.geometry.TriangleMesh()
        for array in array:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)  # create a small sphere to represent point
            sphere.translate(array)  # translate this sphere to point
            geometries += sphere
        geometries.paint_uniform_color([1, 0.1, 0.5])
        return geometries

    highlight_pnts = create_geometry_at_points(mesh2.points)
    # o3d.visualization.draw_geometries([highlight_pnts],
    #                                    window_name="Landmarks and Face Mesh",
    #                                    width = 800, height = 800, left=800, top=200)

    # Get the landmarks as numpy array
    landmarks = np.asarray(mesh2.points)

    # Define the bounding box around the landmarksq
    min_bound = np.min(landmarks, axis=0)
    max_bound = np.max(landmarks, axis=0)

    # Create an AxisAlignedBoundingBox
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    # Crop the face region from the original mesh using the bounding box
    face_region = body.crop(bbox)
    o3d.io.write_point_cloud("../outputs/3d_models/face_region.ply", face_region)
    mesh2 = o3d.io.read_point_cloud("../outputs/3d_models/face_region.ply")

    # Visualize the cropped face region
    # o3d.visualization.draw_geometries([mesh2], window_name="Face Region",
    #                                width=800, height=800, left=800, top=200)

    # Estimate normals
    mesh2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    poisson_mesh = \
    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(mesh2, depth=9, width=0, scale=2, linear_fit=False)[0]

    bbox = mesh2.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    def lod_mesh_export(p_mesh_crop, lods, extension, path):
        mesh_lods = {}
        for i in lods:
            mesh_lod = p_mesh_crop.simplify_quadric_decimation(i)
            o3d.io.write_triangle_mesh(path + "Face" + extension, mesh_lod)
            mesh_lods[i] = mesh_lod
        print("generation of " + str(i) + " LoD successful")
        return mesh_lods

    output_path = "../outputs/3d_models/"
    my_lods = lod_mesh_export(p_mesh_crop, [len(mesh2.points)], ".ply", output_path)

    Face = o3d.io.read_point_cloud("../outputs/3d_models/Face.ply")
    # Visualize the cropped face region
    # o3d.visualization.draw_geometries([Face], window_name="Face Region",
    #                                    width=800, height=800, left=800, top=200)

    # Define indices for left and right facial landmarks
    # Extract left and right facial landmarks
    left_landmarks = array[:247]
    # print(len(left_indices))
    right_landmarks = array[248:]
    # print(right_indices)

    # Create PointCloud objects for left and right landmarks
    left_pcd = o3d.geometry.PointCloud()
    left_pcd.points = o3d.utility.Vector3dVector(left_landmarks)
    left_pcd.paint_uniform_color([1, 0, 0])  # Red color for left landmarks

    right_pcd = o3d.geometry.PointCloud()
    right_pcd.points = o3d.utility.Vector3dVector(right_landmarks)
    right_pcd.paint_uniform_color([0, 0, 1])  # Blue color for right landmarks

    # Visualize the combined facial landmarks along with the 3D head model
    # o3d.visualization.draw_geometries([left_pcd,right_pcd],
    #                                window_name="Facial Landmarks", width=800, height=600)

    # Get the landmarks as numpy array
    landmarks_left = np.asarray(left_pcd.points)
    landmarks_right = np.asarray(right_pcd.points)

    # Define the bounding box around the landmarksq
    min_bound_left = np.min(landmarks_left, axis=0)
    max_bound_left = np.max(landmarks_left, axis=0)

    # Define the bounding box around the landmarksq
    min_bound_right = np.min(landmarks_right, axis=0)
    max_bound_right = np.max(landmarks_right, axis=0)

    # Create an AxisAlignedBoundingBox
    bbox_left = o3d.geometry.AxisAlignedBoundingBox(min_bound_left, max_bound_left)
    bbox_right = o3d.geometry.AxisAlignedBoundingBox(min_bound_right, max_bound_right)

    # Crop the face region from the original mesh using the bounding box
    face_region_left = Face.crop(bbox_left)
    face_region_right = Face.crop(bbox_right)

    o3d.io.write_point_cloud("../outputs/3d_models/leftface_region.ply", face_region_left)
    o3d.io.write_point_cloud("../outputs/3d_models/rightface_region.ply", face_region_right)
    mesh2 = o3d.io.read_point_cloud("../outputs/3d_models/leftface_region.ply")
    mesh3 = o3d.io.read_point_cloud("../outputs/3d_models/rightface_region.ply")
    # print(dir(mesh2))

    # Visualize the cropped face region
    # o3d.visualization.draw_geometries([mesh2], window_name="Left Face Region",
    #                                width=800, height=800, left=800, top=200)

    # Visualize the cropped face region
    # o3d.visualization.draw_geometries([mesh3], window_name="Right Face Region",
    #                                width=800, height=800, left=800, top=200)


def LandmarksFacee():
    print("entered")
    # Read the point cloud
    pcd = o3d.io.read_point_cloud("../resources/3d_models/Individual1.ply")
    mesh_array = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    points = mesh_array[mesh_array[:, 2] < 650]
    colors1 = colors[mesh_array[:, 2] < 650]

    with open("head.txt", mode='w') as f:
        for i in range(len(points)):
            f.write("%f    " % float(points[i][0].item()))
            f.write("%f    " % float(points[i][1].item()))
            f.write("%f    \n" % float(points[i][2].item()))

    with open("../outputs/text_files/colors1.txt", mode='w') as f:
        for i in range(len(colors1)):
            f.write("%f    " % float(colors1[i][0].item()))
            f.write("%f    " % float(colors1[i][1].item()))
            f.write("%f    \n" % float(colors1[i][2].item()))

    data = np.loadtxt("../outputs/text_files/head.txt")
    colors1 = np.loadtxt("../outputs/text_files/colors1.txt")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.colors = o3d.utility.Vector3dVector(colors1)

    o3d.io.write_point_cloud("../outputs/3d_models/3D_Head.ply", pcd)
    pcd_2 = o3d.io.read_point_cloud("../outputs/3d_models/3D_Head.ply")

    # Rotate to the Front side
    mesh_r = copy.deepcopy(pcd_2)
    R_front = pcd_2.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
    mesh_r.rotate(R_front, center=(0, 0, 0))

    # o3d.visualization.draw_geometries([mesh_r],
    #                                   width=800,height=800,left=50,top=50)

    # remove the outlier by index first time
    voxel_size = 0.05
    pcd_downsampled = mesh_r.voxel_down_sample(voxel_size=voxel_size)
    uni_down_pcd = pcd_downsampled.uniform_down_sample(every_k_points=1)

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        # vis = o3d.visualization.VisualizerWithEditing()
        # vis.create_window(window_name="3D-Head",width = 800, height = 800, left=50, top=50)
        # vis.add_geometry(inlier_cloud)
        # vis.run()  # user picks points
        # vis.capture_screen_image("image-head.jpg", do_render=True)
        # vis.destroy_window()
        # o3d.visualization.draw_geometries([inlier_cloud],width=800,height=800,left=50,top=50)

    print("Statistical oulier removal")
    cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    display_inlier_outlier(uni_down_pcd, ind)
    o3d.io.write_point_cloud("../outputs/3d_models/3D-Head clean.ply", cl)

    # Landmarks 3D face mesh extraction
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    file = open('../outputs/text_files/landmarks.txt', 'w')

    # Load the static image
    image_path = '../outputs/images/image-head.jpg'  # Change this to the path of your image
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
                cv2.circle(annotated_image, landmark_px, 1, (0, 255, 0), -1)
                file.write(f'{landmark_px[0]}\t{landmark_px[1]}\t{(0.5 + landmark.z) * 550}')
                file.write('\n')

    # Close the file
    file.close()

    # Save or display the annotated image
    # cv2.imwrite('annotated_image.jpg', annotated_image)
    # cv2.imshow('Annotated Image', annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    data1 = np.loadtxt("../outputs/text_files/landmarks.txt")
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(data1)
    o3d.io.write_point_cloud("../outputs/3d_models/landmarks.ply", pcd1)

    # Read point clouds
    body = o3d.io.read_point_cloud("../outputs/3d_models/3D-Head clean.ply")
    body1 = o3d.io.read_point_cloud("../outputs/3d_models/landmarks.ply")
    R = body1.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
    body1.rotate(R, center=(0, 0, 0))

    # Get max points
    z_max = max(body.points, key=lambda x: x[2])
    z_max1 = max(body1.points, key=lambda x: x[2])

    # Create sphere at centroid position
    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    centroid_sphere.translate(z_max)

    centroid_sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    centroid_sphere1.translate(z_max1)

    # Create axes LineSets
    def create_axes_line_set(origin):
        axes_line_set = o3d.geometry.LineSet()
        axes_points = np.array([origin, origin + [50, 0, 0], origin + [0, 50, 0], origin + [0, 0, 50]])
        axes_lines = [[0, 1], [0, 2], [0, 3]]
        axes_line_set.points = o3d.utility.Vector3dVector(axes_points)
        axes_line_set.lines = o3d.utility.Vector2iVector(axes_lines)
        axes_line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return axes_line_set

    axes_line_set = create_axes_line_set(z_max)
    axes_line_set1 = create_axes_line_set(z_max1)

    # Calculate translation and scale
    translation = z_max - z_max1
    scale_factor = 1.0  # Adjust as needed

    # Translate and scale landmarks
    body1.translate(translation)
    scaled_landmarks = o3d.geometry.PointCloud()
    scaled_landmarks.points = o3d.utility.Vector3dVector(np.asarray(body1.points) * scale_factor)
    mesh_array = np.asarray(scaled_landmarks.points)
    print(mesh_array)

    with open("../outputs/text_files/scaled_landmarks.txt", mode='w') as f:
        for i in range(len(mesh_array)):
            f.write("%f    " % float(mesh_array[i][0].item()))
            f.write("%f    " % float(mesh_array[i][1].item()))
            f.write("%f    \n" % float(mesh_array[i][2].item()))

    data2 = np.loadtxt("../outputs/text_files/scaled_landmarks.txt")
    pcdext = o3d.io.read_point_cloud("../outputs/text_files/scaled_landmarks.txt", format="xyz")
    o3d.io.write_point_cloud("../outputs/3d_models/scaled_landmarks.ply", pcdext)
    mesh2 = o3d.io.read_point_cloud("../outputs/3d_models/scaled_landmarks.ply")
    array = np.asarray(mesh2.points)

    def create_geometry_at_points(array):
        geometries = o3d.geometry.TriangleMesh()
        for array in array:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)  # create a small sphere to represent point
            sphere.translate(array)  # translate this sphere to point
            geometries += sphere
        geometries.paint_uniform_color([1, 0.1, 0.5])
        return geometries

    highlight_pnts = create_geometry_at_points(mesh2.points)
    # o3d.visualization.draw_geometries([highlight_pnts],
    #                                    window_name="Landmarks and Face Mesh",
    #                                    width = 800, height = 800, left=800, top=200)


def LeftFace():
    print("entered")
    # Read the point cloud
    pcd = o3d.io.read_point_cloud("Individual1.ply")
    mesh_array = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    points = mesh_array[mesh_array[:, 2] < 650]
    colors1 = colors[mesh_array[:, 2] < 650]

    with open("../outputs/text_files/head.txt", mode='w') as f:
        for i in range(len(points)):
            f.write("%f    " % float(points[i][0].item()))
            f.write("%f    " % float(points[i][1].item()))
            f.write("%f    \n" % float(points[i][2].item()))

    with open("../outputs/text_files/colors1.txt", mode='w') as f:
        for i in range(len(colors1)):
            f.write("%f    " % float(colors1[i][0].item()))
            f.write("%f    " % float(colors1[i][1].item()))
            f.write("%f    \n" % float(colors1[i][2].item()))

    data = np.loadtxt("../outputs/text_files/head.txt")
    colors1 = np.loadtxt("../outputs/text_files/colors1.txt")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.colors = o3d.utility.Vector3dVector(colors1)

    o3d.io.write_point_cloud("../outputs/3d_models/3D_Head.ply", pcd)
    pcd_2 = o3d.io.read_point_cloud("../outputs/3d_models/3D_Head.ply")

    # Rotate to the Front side
    mesh_r = copy.deepcopy(pcd_2)
    R_front = pcd_2.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
    mesh_r.rotate(R_front, center=(0, 0, 0))

    # o3d.visualization.draw_geometries([mesh_r],
    #                                   width=800,height=800,left=50,top=50)

    # remove the outlier by index first time
    voxel_size = 0.05
    pcd_downsampled = mesh_r.voxel_down_sample(voxel_size=voxel_size)
    uni_down_pcd = pcd_downsampled.uniform_down_sample(every_k_points=1)

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        # vis = o3d.visualization.VisualizerWithEditing()
        # vis.create_window(window_name="3D-Head",width = 800, height = 800, left=50, top=50)
        # vis.add_geometry(inlier_cloud)
        # vis.run()  # user picks points
        # vis.capture_screen_image("image-head.jpg", do_render=True)
        # vis.destroy_window()
        # o3d.visualization.draw_geometries([inlier_cloud],width=800,height=800,left=50,top=50)

    print("Statistical oulier removal")
    cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    display_inlier_outlier(uni_down_pcd, ind)
    o3d.io.write_point_cloud("../outputs/3d_models/3D-Head clean.ply", cl)

    # Landmarks 3D face mesh extraction
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    file = open('../outputs/text_files/landmarks.txt', 'w')

    # Load the static image
    image_path = '../outputs/images/image-head.jpg'  # Change this to the path of your image
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
                cv2.circle(annotated_image, landmark_px, 1, (0, 255, 0), -1)
                file.write(f'{landmark_px[0]}\t{landmark_px[1]}\t{(0.5 + landmark.z) * 550}')
                file.write('\n')

    # Close the file
    file.close()

    # Save or display the annotated image
    # cv2.imwrite('annotated_image.jpg', annotated_image)
    # cv2.imshow('Annotated Image', annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    data1 = np.loadtxt("../outputs/text_files/landmarks.txt")
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(data1)
    o3d.io.write_point_cloud("../outputs/3d_models/landmarks.ply", pcd1)

    # Read point clouds
    body = o3d.io.read_point_cloud("../outputs/3d_models/3D-Head clean.ply")
    body1 = o3d.io.read_point_cloud("../outputs/3d_models/landmarks.ply")
    R = body1.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
    body1.rotate(R, center=(0, 0, 0))

    # Get max points
    z_max = max(body.points, key=lambda x: x[2])
    z_max1 = max(body1.points, key=lambda x: x[2])

    # Create sphere at centroid position
    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    centroid_sphere.translate(z_max)

    centroid_sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    centroid_sphere1.translate(z_max1)

    # Create axes LineSets
    def create_axes_line_set(origin):
        axes_line_set = o3d.geometry.LineSet()
        axes_points = np.array([origin, origin + [50, 0, 0], origin + [0, 50, 0], origin + [0, 0, 50]])
        axes_lines = [[0, 1], [0, 2], [0, 3]]
        axes_line_set.points = o3d.utility.Vector3dVector(axes_points)
        axes_line_set.lines = o3d.utility.Vector2iVector(axes_lines)
        axes_line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return axes_line_set

    axes_line_set = create_axes_line_set(z_max)
    axes_line_set1 = create_axes_line_set(z_max1)

    # Calculate translation and scale
    translation = z_max - z_max1
    scale_factor = 1.0  # Adjust as needed

    # Translate and scale landmarks
    body1.translate(translation)
    scaled_landmarks = o3d.geometry.PointCloud()
    scaled_landmarks.points = o3d.utility.Vector3dVector(np.asarray(body1.points) * scale_factor)
    mesh_array = np.asarray(scaled_landmarks.points)
    print(mesh_array)

    with open("../outputs/text_files/scaled_landmarks.txt", mode='w') as f:
        for i in range(len(mesh_array)):
            f.write("%f    " % float(mesh_array[i][0].item()))
            f.write("%f    " % float(mesh_array[i][1].item()))
            f.write("%f    \n" % float(mesh_array[i][2].item()))

    data2 = np.loadtxt("../outputs/text_files/scaled_landmarks.txt")
    pcdext = o3d.io.read_point_cloud("../outputs/text_files/scaled_landmarks.txt", format="xyz")
    o3d.io.write_point_cloud("../outputs/3d_models/scaled_landmarks.ply", pcdext)
    mesh2 = o3d.io.read_point_cloud("../outputs/3d_models/scaled_landmarks.ply")
    array = np.asarray(mesh2.points)

    def create_geometry_at_points(array):
        geometries = o3d.geometry.TriangleMesh()
        for array in array:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)  # create a small sphere to represent point
            sphere.translate(array)  # translate this sphere to point
            geometries += sphere
        geometries.paint_uniform_color([1, 0.1, 0.5])
        return geometries

    highlight_pnts = create_geometry_at_points(mesh2.points)
    # o3d.visualization.draw_geometries([highlight_pnts],
    #                                    window_name="Landmarks and Face Mesh",
    #                                    width = 800, height = 800, left=800, top=200)

    # Get the landmarks as numpy array
    landmarks = np.asarray(mesh2.points)

    # Define the bounding box around the landmarksq
    min_bound = np.min(landmarks, axis=0)
    max_bound = np.max(landmarks, axis=0)

    # Create an AxisAlignedBoundingBox
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    # Crop the face region from the original mesh using the bounding box
    face_region = body.crop(bbox)
    o3d.io.write_point_cloud("../outputs/3d_models/face_region.ply", face_region)
    mesh2 = o3d.io.read_point_cloud("../outputs/3d_models/face_region.ply")

    # Visualize the cropped face region
    # o3d.visualization.draw_geometries([mesh2], window_name="Face Region",
    #                                width=800, height=800, left=800, top=200)

    # Estimate normals
    mesh2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    poisson_mesh = \
    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(mesh2, depth=9, width=0, scale=2, linear_fit=False)[0]

    bbox = mesh2.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    def lod_mesh_export(p_mesh_crop, lods, extension, path):
        mesh_lods = {}
        for i in lods:
            mesh_lod = p_mesh_crop.simplify_quadric_decimation(i)
            o3d.io.write_triangle_mesh(path + "Face" + extension, mesh_lod)
            mesh_lods[i] = mesh_lod
        print("generation of " + str(i) + " LoD successful")
        return mesh_lods

    output_path = ""
    my_lods = lod_mesh_export(p_mesh_crop, [len(mesh2.points)], ".ply", output_path)

    Face = o3d.io.read_point_cloud("../outputs/3d_models/Face.ply")
    # Visualize the cropped face region
    o3d.visualization.draw_geometries([Face], window_name="Face Region",
                                      width=800, height=800, left=800, top=200)

    # Define indices for left and right facial landmarks
    # Extract left and right facial landmarks
    left_landmarks = array[:247]
    # print(len(left_indices))
    right_landmarks = array[248:]
    # print(right_indices)

    # Create PointCloud objects for left and right landmarks
    left_pcd = o3d.geometry.PointCloud()
    left_pcd.points = o3d.utility.Vector3dVector(left_landmarks)
    left_pcd.paint_uniform_color([1, 0, 0])  # Red color for left landmarks

    right_pcd = o3d.geometry.PointCloud()
    right_pcd.points = o3d.utility.Vector3dVector(right_landmarks)
    right_pcd.paint_uniform_color([0, 0, 1])  # Blue color for right landmarks

    # Visualize the combined facial landmarks along with the 3D head model
    # o3d.visualization.draw_geometries([left_pcd,right_pcd],
    #                                window_name="Facial Landmarks", width=800, height=600)

    # Get the landmarks as numpy array
    landmarks_left = np.asarray(left_pcd.points)
    landmarks_right = np.asarray(right_pcd.points)

    # Define the bounding box around the landmarksq
    min_bound_left = np.min(landmarks_left, axis=0)
    max_bound_left = np.max(landmarks_left, axis=0)

    # Define the bounding box around the landmarksq
    min_bound_right = np.min(landmarks_right, axis=0)
    max_bound_right = np.max(landmarks_right, axis=0)

    # Create an AxisAlignedBoundingBox
    bbox_left = o3d.geometry.AxisAlignedBoundingBox(min_bound_left, max_bound_left)
    bbox_right = o3d.geometry.AxisAlignedBoundingBox(min_bound_right, max_bound_right)

    # Crop the face region from the original mesh using the bounding box
    face_region_left = Face.crop(bbox_left)
    face_region_right = Face.crop(bbox_right)

    o3d.io.write_point_cloud("../outputs/3d_models/leftface_region.ply", face_region_left)
    o3d.io.write_point_cloud("../outputs/3d_models/rightface_region.ply", face_region_right)
    mesh2 = o3d.io.read_point_cloud("../outputs/3d_models/leftface_region.ply")
    mesh3 = o3d.io.read_point_cloud("../outputs/3d_models/rightface_region.ply")
    # print(dir(mesh2))

    # Visualize the cropped face region
    # o3d.visualization.draw_geometries([mesh2], window_name="Left Face Region",
    #                                width=800, height=800, left=800, top=200)

    # Visualize the cropped face region
    # o3d.visualization.draw_geometries([mesh3], window_name="Right Face Region",
    #                                width=800, height=800, left=800, top=200)


def RightFace():
    print("entered")
    # Read the point cloud
    pcd = o3d.io.read_point_cloud("../resources/3d_models/Individual1.ply")
    mesh_array = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    points = mesh_array[mesh_array[:, 2] < 650]
    colors1 = colors[mesh_array[:, 2] < 650]

    with open("../outputs/text_files/head.txt", mode='w') as f:
        for i in range(len(points)):
            f.write("%f    " % float(points[i][0].item()))
            f.write("%f    " % float(points[i][1].item()))
            f.write("%f    \n" % float(points[i][2].item()))

    with open("../outputs/text_files/colors1.txt", mode='w') as f:
        for i in range(len(colors1)):
            f.write("%f    " % float(colors1[i][0].item()))
            f.write("%f    " % float(colors1[i][1].item()))
            f.write("%f    \n" % float(colors1[i][2].item()))

    data = np.loadtxt("../outputs/text_files/head.txt")
    colors1 = np.loadtxt("../outputs/text_files/colors1.txt")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.colors = o3d.utility.Vector3dVector(colors1)

    o3d.io.write_point_cloud("../outputs/3d_models/3D_Head.ply", pcd)
    pcd_2 = o3d.io.read_point_cloud("../outputs/3d_models/3D_Head.ply")

    # Rotate to the Front side
    mesh_r = copy.deepcopy(pcd_2)
    R_front = pcd_2.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
    mesh_r.rotate(R_front, center=(0, 0, 0))

    # o3d.visualization.draw_geometries([mesh_r],
    #                                   width=800,height=800,left=50,top=50)

    # remove the outlier by index first time
    voxel_size = 0.05
    pcd_downsampled = mesh_r.voxel_down_sample(voxel_size=voxel_size)
    uni_down_pcd = pcd_downsampled.uniform_down_sample(every_k_points=1)

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        # vis = o3d.visualization.VisualizerWithEditing()
        # vis.create_window(window_name="3D-Head",width = 800, height = 800, left=50, top=50)
        # vis.add_geometry(inlier_cloud)
        # vis.run()  # user picks points
        # vis.capture_screen_image("image-head.jpg", do_render=True)
        # vis.destroy_window()
        # o3d.visualization.draw_geometries([inlier_cloud],width=800,height=800,left=50,top=50)

    print("Statistical oulier removal")
    cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    display_inlier_outlier(uni_down_pcd, ind)
    o3d.io.write_point_cloud("../outputs/3d_models/3D-Head clean.ply", cl)

    # Landmarks 3D face mesh extraction
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    file = open('../outputs/text_files/landmarks.txt', 'w')

    # Load the static image
    image_path = '../outputs/images/image-head.jpg'  # Change this to the path of your image
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
                cv2.circle(annotated_image, landmark_px, 1, (0, 255, 0), -1)
                file.write(f'{landmark_px[0]}\t{landmark_px[1]}\t{(0.5 + landmark.z) * 550}')
                file.write('\n')

    # Close the file
    file.close()

    # Save or display the annotated image
    # cv2.imwrite('annotated_image.jpg', annotated_image)
    # cv2.imshow('Annotated Image', annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    data1 = np.loadtxt("../outputs/text_files/landmarks.txt")
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(data1)
    o3d.io.write_point_cloud("../outputs/3d_models/landmarks.ply", pcd1)

    # Read point clouds
    body = o3d.io.read_point_cloud("../outputs/3d_models/3D-Head clean.ply")
    body1 = o3d.io.read_point_cloud("../outputs/3d_models/landmarks.ply")
    R = body1.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
    body1.rotate(R, center=(0, 0, 0))

    # Get max points
    z_max = max(body.points, key=lambda x: x[2])
    z_max1 = max(body1.points, key=lambda x: x[2])

    # Create sphere at centroid position
    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    centroid_sphere.translate(z_max)

    centroid_sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    centroid_sphere1.translate(z_max1)

    # Create axes LineSets
    def create_axes_line_set(origin):
        axes_line_set = o3d.geometry.LineSet()
        axes_points = np.array([origin, origin + [50, 0, 0], origin + [0, 50, 0], origin + [0, 0, 50]])
        axes_lines = [[0, 1], [0, 2], [0, 3]]
        axes_line_set.points = o3d.utility.Vector3dVector(axes_points)
        axes_line_set.lines = o3d.utility.Vector2iVector(axes_lines)
        axes_line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return axes_line_set

    axes_line_set = create_axes_line_set(z_max)
    axes_line_set1 = create_axes_line_set(z_max1)

    # Calculate translation and scale
    translation = z_max - z_max1
    scale_factor = 1.0  # Adjust as needed

    # Translate and scale landmarks
    body1.translate(translation)
    scaled_landmarks = o3d.geometry.PointCloud()
    scaled_landmarks.points = o3d.utility.Vector3dVector(np.asarray(body1.points) * scale_factor)
    mesh_array = np.asarray(scaled_landmarks.points)
    print(mesh_array)

    with open("../outputs/text_files/scaled_landmarks.txt", mode='w') as f:
        for i in range(len(mesh_array)):
            f.write("%f    " % float(mesh_array[i][0].item()))
            f.write("%f    " % float(mesh_array[i][1].item()))
            f.write("%f    \n" % float(mesh_array[i][2].item()))

    data2 = np.loadtxt("../outputs/text_files/scaled_landmarks.txt")
    pcdext = o3d.io.read_point_cloud("../outputs/text_files/scaled_landmarks.txt", format="xyz")
    o3d.io.write_point_cloud("../outputs/3d_models/scaled_landmarks.ply", pcdext)
    mesh2 = o3d.io.read_point_cloud("../outputs/3d_models/scaled_landmarks.ply")
    array = np.asarray(mesh2.points)

    def create_geometry_at_points(array):
        geometries = o3d.geometry.TriangleMesh()
        for array in array:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)  # create a small sphere to represent point
            sphere.translate(array)  # translate this sphere to point
            geometries += sphere
        geometries.paint_uniform_color([1, 0.1, 0.5])
        return geometries

    highlight_pnts = create_geometry_at_points(mesh2.points)
    # o3d.visualization.draw_geometries([highlight_pnts],
    #                                    window_name="Landmarks and Face Mesh",
    #                                    width = 800, height = 800, left=800, top=200)

    # Get the landmarks as numpy array
    landmarks = np.asarray(mesh2.points)

    # Define the bounding box around the landmarksq
    min_bound = np.min(landmarks, axis=0)
    max_bound = np.max(landmarks, axis=0)

    # Create an AxisAlignedBoundingBox
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    # Crop the face region from the original mesh using the bounding box
    face_region = body.crop(bbox)
    o3d.io.write_point_cloud("../outputs/3d_models/face_region.ply", face_region)
    mesh2 = o3d.io.read_point_cloud("../outputs/3d_models/face_region.ply")

    # Visualize the cropped face region
    # o3d.visualization.draw_geometries([mesh2], window_name="Face Region",
    #                                width=800, height=800, left=800, top=200)

    # Estimate normals
    mesh2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    poisson_mesh = \
    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(mesh2, depth=9, width=0, scale=2, linear_fit=False)[0]

    bbox = mesh2.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    def lod_mesh_export(p_mesh_crop, lods, extension, path):
        mesh_lods = {}
        for i in lods:
            mesh_lod = p_mesh_crop.simplify_quadric_decimation(i)
            o3d.io.write_triangle_mesh(path + "Face" + extension, mesh_lod)
            mesh_lods[i] = mesh_lod
        print("generation of " + str(i) + " LoD successful")
        return mesh_lods

    output_path = ""
    my_lods = lod_mesh_export(p_mesh_crop, [len(mesh2.points)], ".ply", output_path)

    Face = o3d.io.read_point_cloud("../outputs/3d_models/Face.ply")
    # Visualize the cropped face region
    o3d.visualization.draw_geometries([Face], window_name="Face Region",
                                      width=800, height=800, left=800, top=200)

    # Define indices for left and right facial landmarks
    # Extract left and right facial landmarks
    left_landmarks = array[:247]
    # print(len(left_indices))
    right_landmarks = array[248:]
    # print(right_indices)

    # Create PointCloud objects for left and right landmarks
    left_pcd = o3d.geometry.PointCloud()
    left_pcd.points = o3d.utility.Vector3dVector(left_landmarks)
    left_pcd.paint_uniform_color([1, 0, 0])  # Red color for left landmarks

    right_pcd = o3d.geometry.PointCloud()
    right_pcd.points = o3d.utility.Vector3dVector(right_landmarks)
    right_pcd.paint_uniform_color([0, 0, 1])  # Blue color for right landmarks

    # Visualize the combined facial landmarks along with the 3D head model
    # o3d.visualization.draw_geometries([left_pcd,right_pcd],
    #                                window_name="Facial Landmarks", width=800, height=600)

    # Get the landmarks as numpy array
    landmarks_left = np.asarray(left_pcd.points)
    landmarks_right = np.asarray(right_pcd.points)

    # Define the bounding box around the landmarksq
    min_bound_left = np.min(landmarks_left, axis=0)
    max_bound_left = np.max(landmarks_left, axis=0)

    # Define the bounding box around the landmarksq
    min_bound_right = np.min(landmarks_right, axis=0)
    max_bound_right = np.max(landmarks_right, axis=0)

    # Create an AxisAlignedBoundingBox
    bbox_left = o3d.geometry.AxisAlignedBoundingBox(min_bound_left, max_bound_left)
    bbox_right = o3d.geometry.AxisAlignedBoundingBox(min_bound_right, max_bound_right)

    # Crop the face region from the original mesh using the bounding box
    face_region_left = Face.crop(bbox_left)
    face_region_right = Face.crop(bbox_right)

    o3d.io.write_point_cloud("../outputs/3d_models/leftface_region.ply", face_region_left)
    o3d.io.write_point_cloud("../outputs/3d_models/rightface_region.ply", face_region_right)
    mesh2 = o3d.io.read_point_cloud("../outputs/3d_models/leftface_region.ply")
    mesh3 = o3d.io.read_point_cloud("../outputs/3d_models/rightface_region.ply")
    # print(dir(mesh2))

    # Visualize the cropped face region
    # o3d.visualization.draw_geometries([mesh2], window_name="Left Face Region",
    #                                width=800, height=800, left=800, top=200)

    # Visualize the cropped face region
    # o3d.visualization.draw_geometries([mesh3], window_name="Right Face Region",
    #                                width=800, height=800, left=800, top=200)