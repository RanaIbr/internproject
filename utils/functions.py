from facial3dprocessingf import *
from tkinter import messagebox

def headExtraction3d(path):
    print("entered")
    # Read the point cloud
    pcd = o3d.io.read_point_cloud(path)
    mesh_array = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    points = mesh_array[mesh_array[:, 2] < 650]
    colors1 = colors[mesh_array[:, 2] < 650]

    with open("head.txt", mode='w') as f:
        for i in range(len(points)):
            f.write("%f    " % float(points[i][0].item()))
            f.write("%f    " % float(points[i][1].item()))
            f.write("%f    \n" % float(points[i][2].item()))

    with open("colors1.txt", mode='w') as f:
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

    cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    inlier_cloud = uni_down_pcd.select_by_index(ind)
    outlier_cloud = uni_down_pcd.select_by_index(ind, invert=True)
    o3d.io.write_point_cloud("../outputs/3d_models/3D_Head clean.ply", inlier_cloud)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="3D-Head", width=800, height=800, left=50, top=50)
    vis.add_geometry(inlier_cloud)
    vis.run()  # user picks points
    vis.capture_screen_image("../outputs/images/image-head.jpg", do_render=True)
    vis.destroy_window()

    # # o3d.io.write_point_cloud("3D-Head clean.ply", cl)
    # self.load('3D_Head clean.ply')


def faceMeshLandmarks3d(path_):
    if not path_:
        messagebox.showerror("Error", "Please select a file")
    else:
        facial()
    # self.load('scaled_landmarks.ply')


def faceExtraction3d(path_):
    if not path_:
        messagebox.showerror("Error", "Please select a file")
    else:
        Face = o3d.io.read_point_cloud("../outputs/3d_models/Face.ply")
        o3d.visualization.draw_geometries([Face], window_name="Face Region",
                                           width=800, height=800, left=800, top=200)


def leftFaceRegion(path_):
    if not path_:
        messagebox.showerror("Error", "Please select a file")
    else:
        print("leftFaceRegion")
     # LeftFace()
        Face = o3d.io.read_point_cloud("../outputs/3d_models/leftface_region.ply")
        o3d.visualization.draw_geometries([Face], window_name="Left Face Region",
                                           width=800, height=800, left=800, top=200)

def rightFaceRegion(path_):
    if not path_:
        messagebox.showerror("Error", "Please select a file")
    else:
        print("rightFaceRegion")
      #  RightFace()
        # self.load('rightface_region.ply')
        Face = o3d.io.read_point_cloud("../outputs/3d_models/rightface_region.ply")
        o3d.visualization.draw_geometries([Face], window_name="Right Face Region",
                                           width=800, height=800, left=800, top=200)


def landmarks2d(path_):
    if not path_:
        messagebox.showerror("Error", "Please select a file")
    else:
        imageLandmarks()


def imageLandmarks(path_):
    if not path_:
        messagebox.showerror("Error", "Please select a file")
    else:
        body = o3d.io.read_point_cloud("../outputs/3d_models/cropped_1.ply")

    def front(body):
        if not path_:
            messagebox.showerror("Error", "Please select a file")
        else:
            print("front")
            # Front body display
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name="Front side", width=800, height=800, left=800, top=200)
            vis.add_geometry(body)
            vis.run()  # User picks points
            vis.capture_screen_image("../outputs/images/image-body-front.jpg", do_render=True)
            vis.destroy_window()

    front(body)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

    # Read the front and back images
    image_front = cv2.imread('../outputs/images/image-body-front.jpg')

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
    def detectSelectedPoseWithNames(image, pose, selected_indices, landmark_names,path_):
        if not path_:
            messagebox.showerror("Error", "Please select a file")
        else:
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
            # cv2.imshow("Image with Selected Landmarks", output_image)
            cv2.imwrite('../outputs/images/Frontimage.jpg', output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return landmarks_2d

    # Detect and extract 2D landmarks from front and back images
    # landmarks_back = detectSelectedPoseWithNames(image_back, pose, selected_indices, selected_landmark_names)
    landmarks_front = detectSelectedPoseWithNames(image_front, pose, selected_indices, selected_landmark_names)


def cleanAndBodyShape(path):
    if not path:
        messagebox.showerror("Error", "Please select a file")
    else:
        pcd = o3d.io.read_point_cloud(path)
        # vis = o3d.visualization.VisualizerWithEditing()
        # vis.create_window(window_name="Full Body", width=800, height=800, left=800, top=200)
        # vis.add_geometry(pcd)
        # vis.run()
        # vis.destroy_window()
        # rotate to the Front side
        mesh_r = copy.deepcopy(pcd)
        R = pcd.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
        mesh_r.rotate(R, center=(0, 0, 0))

        # Remove the table
        # Define the cropping box of the table
        min_bound = np.array([50.5, -2713.3, 50.5])
        max_bound = np.array([1671.3, -2070.3, 2513.3])

        # Filter points within the bounding box
        filtered_indices = np.all((mesh_r.points >= min_bound) & (mesh_r.points <= max_bound), axis=1)
        cropped_point_cloud = mesh_r.select_by_index(np.where(~filtered_indices)[0])

        # save to .ply
        o3d.io.write_point_cloud("../outputs/3d_models/removetable.ply", cropped_point_cloud)
        mesh = o3d.io.read_point_cloud("../outputs/3d_models/removetable.ply")
        # o3d.visualization.draw_geometries([cropped_point_cloud],window_name="Without Table",width=800,height=800,left=50,top=50)

        voxel_size = 0.05
        pcd_downsampled = cropped_point_cloud.voxel_down_sample(voxel_size=voxel_size)
        uni_down_pcd = pcd_downsampled.uniform_down_sample(every_k_points=4)
        print("871")

        # o3d.visualization.draw_geometries([uni_down_pcd],width=800,height=800,left=50,top=50)

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        # outlier_cloud.paint_uniform_color([1, 0, 0])
        # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        # o3d.visualization.draw_geometries([inlier_cloud],width=800,height=800,left=50,top=50)

    print("Statistical oulier removal")
    cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    display_inlier_outlier(uni_down_pcd, ind)

    # apply final filter poisson
    distances = cl.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    print(avg_dist)
    print(radius)

    poisson_mesh = \
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cl, depth=9, width=0, scale=2, linear_fit=False)[
            0]
    bbox = cl.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    print("900")

    def lod_mesh_export(p_mesh_crop, lods, extension, path):
        mesh_lods = {}
        for i in lods:
            mesh_lod = p_mesh_crop.simplify_quadric_decimation(i)
            o3d.io.write_triangle_mesh(path + "clean" + extension, mesh_lod)
            mesh_lods[i] = mesh_lod
        print("generation of " + str(i) + " LoD successful")
        return mesh_lods

    output_path = "../outputs/3d_models/"
    my_lods = lod_mesh_export(p_mesh_crop, [len(pcd.points)], ".ply", output_path)

    pcd = o3d.io.read_point_cloud("../outputs/3d_models/clean.ply")

    # path = 'outputs/3d_models/clean.ply'
    # self.load(path)
    # print("load")


def geometryCropping(path_):
    if not path_:
        messagebox.showerror("Error", "Please select a file")
    else:
        path = '../outputs/3d_models/clean.ply'
        body = o3d.io.read_point_cloud(path)
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Full Body", width=800, height=800, left=800, top=200)
        vis.add_geometry(body)
        vis.run()
        vis.destroy_window()
        # path = 'outputs/3d_models/cropped_1.ply'
        #
        # self.load(path)

#
# cleanAndBodyShape()
# geometryCropping()
# headExtraction3d()
# faceMeshLandmarks3d()
# faceExtraction3d()
# leftFaceRegion()
# rightFaceRegion()
# landmarks2d()