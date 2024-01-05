import base64
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from .utils import *
from .captureFront2d import *
from .alldistances import *
# from finalResult import *
from .code1 import *
from .captureBack2d import *
from .backdistance import *
from .frontdistance import *
import time
from .automatic_landmarks import *
import os
import open3d as o3d
import glob
# import pdfkit
from tkhtmlview import HTMLLabel
import pyvista as pv
# import tkinter as tk
from tkinter import messagebox
import aspose.threed as a3d
import requests
import json
import ttkbootstrap as tk


class MainWindow3D(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent
        # self.initUI()
        # self.attributes("-zoomed", True)
        self.geometry("1400x750")
        self.title("3D Full Body Human Viewer")

        self.create_buttons()

        self.path = ""
        self.file_path = ""
        self.path_stl = ""
        self.file_path_stl = ""
        self.html_content = ""
        self.html_label = ""

    def create_buttons(self):
        menubar = Menu(self)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Select File", command=self.showDialog)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

        self.current_file = None
        self.Clean3DPLY = None
        # Check if file exists
        if os.path.isfile('resources/3d_models/cleanBody.ply'):
            self.Clean3DPLY = 'resources/3d_models/cleanBody.ply'

        # Create the main frame
        main_frame = Frame(self)
        main_frame.pack(side="top", pady=10, anchor="w")

        giud_text = "Demo for manual geometry cropping \n 1) Press 'Y' twice to align geometry with negative " \
                    "direction of y-axis \n 2) Press 'K' to lock screen and to switch to selection mode \n 3) " \
                    "Drag for rectangle selection, \n or use ctrl + left click for polygon selection \n 4) Press " \
                    "'C' to get a selected geometry and to save it \n 5) Press 'F' to switch to freeview mode"

        self.guide = Label(main_frame, text=giud_text)
        self.guide.pack(side="right", padx=10, pady=10)

        # Load and resize the image
        image_path = "resources/images/trans.png"  # Replace with the actual path to your image
        image = Image.open(image_path)
        image = image.resize((400, 500))

        # Create a PhotoImage object from the image and store it as an instance variable
        self.photo = ImageTk.PhotoImage(image)

        # Create the image label
        self.image_label = Label(main_frame, image=self.photo)
        self.image_label.pack(side="right", padx=10, pady=10)

        image_path_ = "resources/images/logo_19.jpg"  # Replace with the actual path to your image
        image_ = Image.open(image_path_)
        image_ = image_.resize((120, 130))

        # Create a PhotoImage object from the image and store it as an instance variable
        self.photo_ = ImageTk.PhotoImage(image_)

        # Create the image label
        self.image_label_ = Label(main_frame, image=self.photo_)
        self.image_label_.pack(side="top", padx=10, pady=10)

        # Center the image within the top frame
        self.image_label_.pack(anchor="center")

        self.lable = Label(main_frame, text="Select a .ply file to Start !", wraplength=300)
        self.lable.pack(side="top", padx=10, pady=10)

        # Create the top frame for buttons
        top_frame = Frame(main_frame)
        top_frame.pack(side="left", padx=10, pady=10, anchor="w")
        # Load and resize the image

        # Create the bottom frame for lable_distance and label_logo
        bottom_frame = Frame(main_frame)
        bottom_frame.pack(side="right", padx=10, pady=10, anchor="w")
        # Load an image

        button_geo_cropping = Button(top_frame, text="Geometry Cropping", width=25, command=self.demo_3d, height=2)
        button_geo_cropping.pack(side="top", padx=10, pady=5)

        button_clean = Button(top_frame, text="Segmentation", width=25, command=self.clean3d, height=2)
        button_clean.pack(side="top", padx=10, pady=5)

        button_geo_cropping = Button(top_frame, text="2D Automatic Landmarks", width=25, command=self.landmarks2D,
                                     height=2)
        button_geo_cropping.pack(side="top", padx=10, pady=5)

        def finalto():
            allDistancesFunction()
            insert_text()

        button_landmarks_estimation = Button(top_frame, text="3D Manual Landmarks", width=25,
                                             command=finalto, height=2)
        button_landmarks_estimation.pack(side="top", padx=10, pady=5)

        button_slicing = Button(top_frame, text="Slicing", width=25, command=self.slicing, height=2)
        button_slicing.pack(side="top", padx=10, pady=5)

        self.label_logo = Label(top_frame)
        self.label_logo.pack(side="top", padx=10, pady=5)

        # html_content = """
        #    <h1>Hello, HTML in Tkinter!</h1>
        #
        #    """
        #
        #
        # html_label = HTMLLabel(top_frame, html=html_content)
        # html_label.pack(fill="both", expand=True)
        # Function to insert text into the Text widget

        def insert_text():
            # Clear previous content
            text_widget.config(state="normal")
            text_widget.delete('1.0', 'end')
            # Insert new text
            distance_file_path = "outputs/text_files/Frontdistances.txt"
            with open(distance_file_path, 'r') as distance_file:
                distance_text = distance_file.read()

            # Read text from Backdistances.txt
            distance_file_path_back = "outputs/text_files/Backdistances.txt"
            with open(distance_file_path_back, 'r') as distance_file_back:
                distance_text_back = distance_file_back.read()
            text_widget.insert('1.0', distance_text + "\n\n" + distance_text_back)
            text_widget.config(state="disabled")

        button_export = Button(top_frame, text="Export STL", width=25, command=self.exportSTL, height=2)
        button_export.pack(side="top", padx=10, pady=5)

        button_open = Button(top_frame, text="Open STL", width=25, command=self.openSTL, height=2)
        button_open.pack(side="top", padx=10, pady=5)

        button_final = Button(top_frame, text="Final Result", width=25, command=insert_text, height=2)
        button_final.pack(side="top", padx=10, pady=5)

        # button_import = Button(top_frame, text="Import Report", width=25, command=self.ImportReport, height=2)
        # button_import.pack(side="top", padx=10, pady=5)

        button_import = Button(top_frame, text="Upload To Server", width=25, command=self.upload_to_server, height=2)
        button_import.pack(side="top", padx=10, pady=5)

        button_exit = Button(top_frame, text="Exit", command=self.destroy, width=25, height=2)
        button_exit.pack(side="top", padx=10, pady=5)

        # Create a Text widget for the right side
        text_widget = Text(main_frame, width=40, height=30)
        text_widget.pack(side="right", padx=10, pady=10)

    def landmarks2D(self):
        if not (self.file_path):
            messagebox.showerror("Error", "Please select a file")
        else:
            start_front(self.Clean3DPLY)
            start_back(self.Clean3DPLY)

            # Update the guide text
            text = "Click on Landmarks Estimation button then follow these steps:\n\n1) Please pick at least three " \
                   "correspondences using [shift + left click]\nPress [shift + right click] to undo point picking\n" \
                   "2) After picking points, press q to close the window"
            self.guide.config(text=text)

            # Change the image
            # self.change_image("output_image_front.png")

    def change_image(self, path):
        # Load and resize a new image
        new_image_path = path  # Replace with the path to the new image
        new_image = Image.open(new_image_path)
        new_image = new_image.resize((400, 500))

        # Create a new PhotoImage object from the new image and update the instance variable
        self.photo = ImageTk.PhotoImage(new_image)

        # Update the image in the label
        self.image_label.configure(image=self.photo)

    def slicing(self):
        if not (self.file_path):
            messagebox.showerror("Error", "Please select a file")
        else:
            print("urgent:", self.Clean3DPLY)
            body = o3d.io.read_point_cloud(self.Clean3DPLY)
            mesh_array = np.asarray(body.points)
            colors = np.asarray(body.colors)

            mesh_r = copy.deepcopy(body)
            R = body.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
            mesh_r.rotate(R, center=(0, 0, 0))

            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name="Front", width=800, height=900, left=1050, top=100)
            vis.add_geometry(mesh_r)
            vis.run()  # user picks points
            # vis.capture_screen_image("file1.png", do_render=True)
            vis.destroy_window()

            with open("outputs/text_files/arraypoints.txt", mode='w') as f:
                for i in range(len(mesh_array)):
                    f.write("%f    " % float(mesh_array[i][0].item()))
                    f.write("%f    " % float(mesh_array[i][1].item()))
                    f.write("%f    \n" % float(mesh_array[i][2].item()))

            with open("outputs/text_files/colors.txt", mode='w') as f:
                for i in range(len(colors)):
                    f.write("%f    " % float(colors[i][0].item()))
                    f.write("%f    " % float(colors[i][1].item()))
                    f.write("%f    \n" % float(colors[i][2].item()))

            data = np.loadtxt("outputs/text_files/arraypoints.txt")
            colors = np.loadtxt("outputs/text_files/colors.txt")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            o3d.io.write_point_cloud("outputs/3d_models/3D_scanned_data.ply", pcd)

            pcd_2 = o3d.io.read_point_cloud("outputs/3d_models/3D_scanned_data.ply")
            mesh_r1 = copy.deepcopy(pcd_2)
            R = pcd_2.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
            mesh_r1.rotate(R, center=(0, 0, 0))
            #
            # o3d.visualization.draw_geometries([mesh_r1], window_name="Convert", width=800, height=900, left=1050,
            #                                   top=100)

            # normal estimation
            mesh_r1.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            # o3d.visualization.draw_geometries([mesh_r1], window_name="Full body", width=800, height=900, left=1050,
            #                                   top=100)

            array2 = np.asarray(mesh_r1.points)
            colors = np.asarray(mesh_r1.colors)

            points = array2[0:6000, :]
            colors1 = colors[0:6000, :]

            points1 = array2[6000:22000, :]
            colors2 = colors[6000:22000, :]

            points2 = array2[22000:len(array2), :]
            colors3 = colors[22000:len(array2), :]

            print("hon please : ",len(array2))
            with open("outputs/text_files/head.txt", mode='w') as f:
                for i in range(len(points)):
                    f.write("%f    " % float(points[i][0].item()))
                    f.write("%f    " % float(points[i][1].item()))
                    f.write("%f    \n" % float(points[i][2].item()))

            with open("outputs/text_files/colors1.txt", mode='w') as f:
                for i in range(len(colors1)):
                    f.write("%f    " % float(colors1[i][0].item()))
                    f.write("%f    " % float(colors1[i][1].item()))
                    f.write("%f    \n" % float(colors1[i][2].item()))

            data = np.loadtxt("outputs/text_files/head.txt")
            colors1 = np.loadtxt("outputs/text_files/colors1.txt")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data)
            pcd.colors = o3d.utility.Vector3dVector(colors1)

            o3d.io.write_point_cloud("outputs/3d_models/3D_scanned_data.ply", pcd)
            pcd_2 = o3d.io.read_point_cloud("outputs/3d_models/3D_scanned_data.ply")
            o3d.visualization.draw_geometries([pcd_2], window_name="Head", width=800, height=900, left=1050, top=100)

            with open("outputs/text_files/upper body.txt", mode='w') as f:
                for i in range(len(points1)):
                    f.write("%f    " % float(points1[i][0].item()))
                    f.write("%f    " % float(points1[i][1].item()))
                    f.write("%f    \n" % float(points1[i][2].item()))

            with open("outputs/text_files/colors2.txt", mode='w') as f:
                for i in range(len(colors2)):
                    f.write("%f    " % float(colors2[i][0].item()))
                    f.write("%f    " % float(colors2[i][1].item()))
                    f.write("%f    \n" % float(colors2[i][2].item()))

            data1 = np.loadtxt("outputs/text_files/upper body.txt")
            colors2 = np.loadtxt("outputs/text_files/colors2.txt")
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(data1)
            pcd1.colors = o3d.utility.Vector3dVector(colors2)

            o3d.io.write_point_cloud("outputs/3d_models/3D_scanned_data2.ply", pcd1)
            pcd_3 = o3d.io.read_point_cloud("outputs/3d_models/3D_scanned_data2.ply")
            o3d.visualization.draw_geometries([pcd_3], window_name="upper body", width=800, height=900, left=1050,
                                              top=100)

            with open("outputs/text_files/lowerBody.txt", mode='w') as f:
                for i in range(len(points2)):
                    f.write("%f    " % float(points2[i][0].item()))
                    f.write("%f    " % float(points2[i][1].item()))
                    f.write("%f    \n" % float(points2[i][2].item()))

            with open("outputs/text_files/colors3.txt", mode='w') as f:
                for i in range(len(colors3)):
                    f.write("%f    " % float(colors3[i][0].item()))
                    f.write("%f    " % float(colors3[i][1].item()))
                    f.write("%f    \n" % float(colors3[i][2].item()))

            data2 = np.loadtxt("outputs/text_files/lowerBody.txt")
            colors3 = np.loadtxt("outputs/text_files/colors3.txt")
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(data2)
            pcd2.colors = o3d.utility.Vector3dVector(colors3)

            o3d.io.write_point_cloud("outputs/3d_models/3D_scanned_data3.ply", pcd2)
            pcd_4 = o3d.io.read_point_cloud("outputs/3d_models/3D_scanned_data3.ply")

            o3d.visualization.draw_geometries([pcd_4], window_name="lower body", width=800, height=900, left=1050,
                                              top=100)

    def showDialog(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*")])
        if self.file_path:
            self.current_file = self.file_path
            self.lable.config(text="Selected file: " + self.file_path)
            self.path = self.file_path

    def lod_mesh_export(self, p_mesh_crop, lods, extension, path):
        mesh_lods = {}
        for i in lods:
            mesh_lod = p_mesh_crop.simplify_quadric_decimation(i)
            o3d.io.write_triangle_mesh(path + "clean" + extension, mesh_lod)
            mesh_lods[i] = mesh_lod
        print("generation of " + str(i) + " LoD successful")
        return mesh_lods

    def demo_crop_geometry(self, indi):
        print("Demo for manual geometry cropping")
        print(
            "1) Press 'Y' twice to align geometry with negative direction of y-axis"
        )
        print("2) Press 'K' to lock screen and to switch to selection mode")
        print("3) Drag for rectangle selection,")
        print("   or use ctrl + left click for polygon selection")
        print("4) Press 'C' to get a selected geometry and to save it")
        print("5) Press 'F' to switch to freeview mode")
        o3d.visualization.draw_geometries_with_editing([indi], width=800, height=900, left=1050, top=100)

    def upload_to_server(self):
        if not (self.file_path):
            messagebox.showerror("Error", "Please select a file")
        else:
            filename = "outputs/text_files/data.txt"  # Replace with the actual file name

            data = {}  # Dictionary to store the extracted values

            # Open the file in read mode
            with open(filename, "r") as file:
                # Read each line in the file
                for line in file:
                    # Split the line into key and value using the ":" delimiter
                    key, value = line.strip().split(":")
                    # Remove leading/trailing whitespaces from the key and value
                    key = key.strip()
                    value = value.strip()
                    # Store the key-value pair in the data dictionary
                    data[key] = value

            # Extract the values to variables
            name = data.get("name")
            gender = data.get("gender")
            age = int(data.get("age"))
            tall = int(data.get("tall"))
            weight = int(data.get("weight"))
            dimension = data.get("dimension")

            # Read the PDF file as binary data
            pdf_file = 'Final Report.pdf'
            with open(pdf_file, 'rb') as file:
                pdf_data = file.read()

            # Encode the PDF data as base64
            # pdf_base64 = pdf_data.encode('base64')  # For Python 2
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')  # For Python 3

            # Read the image file as binary data
            with open('output_image.jpg', 'rb') as file:
                image_data = file.read()

            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Prepare the payload data as a dictionary
            with open('results.txt', "r") as file:
                file_content = file.read()

            payload = {
                'name': name,
                'gender': gender,
                'age': age,
                'tall': tall,
                'weight': weight,
                'dimension': dimension,
                'landmarks': file_content
            }

            # Read the PDF file as binary data
            with open('Final Report.pdf', 'rb') as file:
                files = {
                    'pdf': file.read()
                }

            # Read the image file as binary data
            with open('output_image.jpg', 'rb') as file:
                files['image'] = file.read()

            # Define the endpoint URL
            url = "https://technologic-lb.com/projectfinalapis/savetoserver.php"

            # Send the POST request with the payload and files
            response = requests.post(url, data=payload, files=files)

            # Check the response status code
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    print("Uploading successful.")
                    messagebox.showinfo("Done !", "Uploading successful.")

                    # Process the response data here
                except json.JSONDecodeError as e:
                    messagebox.showerror("Failed to decode JSON response", e)
                    print("Failed to decode JSON response:", e)
                    print("Response content:", response.content)
            else:
                messagebox.showerror("Failed to upload.", "Status code:" + str(response.status_code))
                print("Failed to upload. Status code:", response.status_code)
                print("Response content:", response.content)

    def demo_3d(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file")
        else:
            # Load the point cloud data from the selected file
            indi = o3d.io.read_point_cloud(self.file_path)

            # Convert point cloud data to numpy arrays for mesh and colors
            mesh_array = np.asarray(indi.points)
            colors = np.asarray(indi.colors)

            # Print mesh and color information for debugging
            print("Mesh Array:", mesh_array)
            print("Colors:", colors)

            # Call the demo_crop_geometry function with the loaded point cloud
            self.demo_crop_geometry(indi)

            # Update the label with the selected file path
            # self.label.config(text=self.path)

            # Set the current file to Clean3DPLY (assuming this is what you intended)
            self.current_file = self.Clean3DPLY

            # Set the guide text
            text = "Click on Clean Of Body"
            self.guide.config(text=text)

    def clean3d(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file")
        else:
            # Read the point cloud from the file
            pcd = o3d.io.read_point_cloud("resources/3d_models/cropped_1.ply")

            # Compute nearest neighbor distances
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 1.5 * avg_dist

            # Print average distance and radius
            print("Average Distance:", avg_dist)
            print("Radius:", radius)

            # Create a Poisson mesh
            poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0, scale=2,
                                                                                     linear_fit=False)[0]

            # Get the bounding box of the point cloud
            bbox = pcd.get_axis_aligned_bounding_box()

            # Crop the Poisson mesh to the bounding box
            p_mesh_crop = poisson_mesh.crop(bbox)

            # Export the cropped mesh
            output_path = "resources/3d_models/"
            my_lods = lod_mesh_export(p_mesh_crop, [len(pcd.points)], ".ply", output_path)

            # Read the cleaned body point cloud
            pcd = o3d.io.read_point_cloud("resources/3d_models/cleanBody.ply")
            # front body display
            mesh_f = copy.deepcopy(pcd)
            R1 = pcd.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
            mesh_f.rotate(R1, center=(0, 0, 0))

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(mesh_f)
            vis.update_geometry(mesh_f)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image("resources/images/screenCleanFront.jpg")
            vis.destroy_window()

            mesh_b = copy.deepcopy(pcd)
            R2 = pcd.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-90)))
            mesh_b.rotate(R2, center=(0, 0, 0))

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(mesh_b)
            vis.update_geometry(mesh_b)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image("resources/images/screenCleanBack.jpg")
            vis.destroy_window()

            # Visualize the cleaned body point cloud
            o3d.visualization.draw_geometries([pcd], width=800, height=900, left=1050, top=100)

            # Update the current file
            self.current_file = self.Clean3DPLY
            print("Clean3DPLY :", self.Clean3DPLY)
            print("current_file :", self.current_file)

    def SelectPoints(self):
        if not (self.file_path):
            messagebox.showerror("Error", "Please select a file")
        else:
            if not (self.Clean3DPLY):
                messagebox.showerror("Error", "Cleaned File Doesnt Exist")
            else:
                # text = "Click on Antropometric Landmarks"
                # self.guide.config(text=text)
                self.change_image("guide_front.jpeg")
                frontLandmarks(self.Clean3DPLY)
                self.change_image("guide_back.jpeg")
                backLandmarks(self.Clean3DPLY)
                self.current_file = self.Clean3DPLY
                self.change_image("resources/images/trans.png")

    def Automatic(self):
        if not (self.file_path):
            messagebox.showerror("Error", "Please select a file")
        else:
            if not (self.Clean3DPLY):
                messagebox.showerror("Error", "Cleaned File Doesnt Exist")
            else:
                startup("resources/3d_models/cleanBody.ply")

    def MarkersPoints(self):
        if not (self.file_path):
            messagebox.showerror("Error", "Please select a file")
        else:
            if not (self.Clean3DPLY):
                messagebox.showerror("Error", "Cleaned File Doesnt Exist")
            else:
                Markers_Points(self.Clean3DPLY)
                # time.sleep(1.5)
                text = "Click on Slicing to Slice the body , the click 'q' on the keyboard"
                self.guide.config(text=text)
                # pcd = o3d.io.read_point_cloud(self.Clean3DPLY)
                self.current_file = self.Clean3DPLY

    def load_image(self):
        image = Image.open("test512.png")
        image = image.resize((100, 100))  # Adjust the size as needed

        self.photo = ImageTk.PhotoImage(image)
        self.label_logo.config(image=self.photo)

    def displayHTML(html_content):
        root = tk.Tk()
        root.title("HTML Display")

        html_label = HTMLLabel(root, html=html_content)
        html_label.pack()

    def distances(self):
        if not (self.file_path):
            messagebox.showerror("Error", "Please select a file")
        else:
            if not (self.Clean3DPLY):
                messagebox.showerror("Error", "Cleaned File Doesnt Exist")
            else:
                dists = GetDistances(self.Clean3DPLY)
                text = ""
                dist1 = round(dists[0])
                dist2 = round(dists[1])
                dist12 = round(dists[2])
                dist3 = round(dists[3])
                dist4 = round(dists[4])
                dist34 = round(dists[5])
                dist5 = round(dists[6])
                dist6 = round(dists[7])
                dist7 = round(dists[8])
                dist67 = round(dists[9])
                dist8 = round(dists[10])
                dist9 = round(dists[11])
                dist89 = round(dists[12])
                dist10 = round(dists[13])
                dist11 = round(dists[14])

                for x in dists:
                    text = "Left upper Arm: " + str(dist1) + " cm" + "\n" + "Left Forearm: " + str(
                        dist2) + " cm" + "\n" + "Left Arm: " + str(dist12) + " cm" + "\n" + "Right upper Arm: " + str(
                        dist3) + " cm" + "\n" + "Right Forearm: " + str(dist4) + " cm" + "\n" + "Right Arm: " + str(
                        dist34) + " cm" + "\n" + "Upper back body: " + str(
                        dist5) + " cm" + "\n" + "Right Thigh: " + str(dist6) + " cm" + "\n" + "Right Shin: " + str(
                        dist7) + " cm" + "\n" + "Right Leg: " + str(dist67) + " cm" + "\n" + "Left Thigh: " + str(
                        dist8) + " cm" + "\n" + "Left Shin: " + str(dist9) + " cm" + "\n" + "Left Leg: " + str(
                        dist89) + " cm" + "\n" + "Full Tall: " + str(dist10) + " cm" + "\n"
                self.guide.config(text=text)

    def openSTL(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("STL Files", "*.stl")])
        if self.file_path:
            if self.file_path.endswith('.stl'):
                self.displaySTL(self.file_path)

    def displaySTL(self, stl_file):
        mesh = pv.read(stl_file)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh)
        plotter.show()

    def showDialog_(self):
        self.file_path_stl = filedialog.askopenfilename(filetypes=[("All Files", "*.*")])
        if self.file_path_stl:
            if (self.file_path.endswith('.stl')):
                self.current_file = self.file_path_stl
                self.lable.config(text="Selected file: " + self.file_path_stl)
                self.path_stl = self.file_path_stl

    def FinalResult(self):
        if not (self.file_path):
            messagebox.showerror("Error", "Please select a file")
        else:
            if not (self.Clean3DPLY):
                messagebox.showerror("Error", "Cleaned File Doesnt Exist")
            else:
                Final_Result(self.Clean3DPLY)
                time.sleep(1.5)
                pcd = o3d.io.read_point_cloud(self.Clean3DPLY)
                # self.filename_label.setText(self.Clean3DPLY)
                self.current_file = self.Clean3DPLY

                dists = GetDistances()
                dist1 = round(dists[0])
                dist2 = round(dists[1])
                dist12 = round(dists[2])
                dist3 = round(dists[3])
                dist4 = round(dists[4])
                dist34 = round(dists[5])
                dist5 = round(dists[6])
                dist6 = round(dists[7])
                dist7 = round(dists[8])
                dist67 = round(dists[9])
                dist8 = round(dists[10])
                dist9 = round(dists[11])
                dist89 = round(dists[12])
                dist10 = round(dists[13])

                for x in dists:
                    text1 = str(dist1) + " cm" + "\n" + str(dist2) + " cm" + "\n" + str(dist12) + " cm" + "\n" + str(
                        dist3) + " cm" + "\n" + str(dist4) + " cm" + "\n" + str(dist34) + " cm" + "\n" + str(
                        dist5) + " cm" + "\n" + str(dist6) + " cm" + "\n" + str(dist7) + " cm" + "\n" + str(
                        dist67) + " cm" + "\n" + str(dist8) + " cm" + "\n" + str(dist9) + " cm" + "\n" + str(
                        dist89) + " cm" + "\n" + str(dist10) + " cm" + "\n"
                self.guide.config(text=text1)

                for x in dists:
                    text2 = "Left upper Arm" + "\n" + "Left Forearm" + "\n" + "Left Arm " + "\n" + "Right upper Arm " + "\n" + "Right Forearm " + "\n" + "Right Arm" + "\n" + "Upper back body " + "\n" + "Right Thigh" + "\n" + "Right Shin" + "\n" + "Right Leg" + "\n" + "Left Thigh" + "\n" + "Left Shin" + "\n" + "Left Leg" + "\n" + "Full Tall" + "\n"
                self.guide.config(text=text2)

                for x in dists:
                    text = "Left upper Arm: " + str(dist1) + " cm" + "\n" + "Left Forearm: " + str(
                        dist2) + " cm" + "\n" + "Left Arm: " + str(dist12) + " cm" + "\n" + "Right upper Arm: " + str(
                        dist3) + " cm" + "\n" + "Right Forearm: " + str(dist4) + " cm" + "\n" + "Right Arm: " + str(
                        dist34) + " cm" + "\n" + "Upper back body: " + str(
                        dist5) + " cm" + "\n" + "Right Thigh: " + str(dist6) + " cm" + "\n" + "Right Shin: " + str(
                        dist7) + " cm" + "\n" + "Right Leg: " + str(dist67) + " cm" + "\n" + "Left Thigh: " + str(
                        dist8) + " cm" + "\n" + "Left Shin: " + str(dist9) + " cm" + "\n" + "Left Leg: " + str(
                        dist89) + " cm" + "\n" + "Full Tall: " + str(dist10) + " cm" + "\n"
                self.guide.config(text=text)

                f = open("results.txt", "w+")
                for i in range(1):
                    f.write(text)
                f.close()

                f = open("results1.txt", "w+")
                for i in range(1):
                    f.write(text2)
                f.close()

                with open('results.txt') as f:
                    lines = f.readlines()

    def exportSTL(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file")
        else:
            if not self.Clean3DPLY:
                messagebox.showerror("Error", "Cleaned File Doesn't Exist")
            else:
                filepath = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL Files", "*.stl")])
                if filepath:
                    scene = a3d.Scene.from_file(self.current_file)
                    scene.save("outputs/3d_models/output_3d_object_for_stl.obj")

                    scene = a3d.Scene.from_file("outputs/3d_models/output_3d_object_for_stl.obj")
                    scene.save(filepath)

    def ImportReport(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file")
        else:
            if not self.Clean3DPLY:
                messagebox.showerror("Error", "Cleaned File Doesn't Exist")
            else:
                # Read text from Frontdistances.txt
                distance_file_path = "outputs/text_files/Frontdistances.txt"
                with open(distance_file_path, 'r') as distance_file:
                    distance_text = distance_file.read()

                # Read text from Frontpoints.txt
                landmark_file_path = "outputs/text_files/Frontpoints.txt"
                with open(landmark_file_path, 'r') as landmark_file:
                    landmark_text = landmark_file.read()

                # Read text from Backdistances.txt
                distance_file_path_back = "outputs/text_files/Backdistances.txt"
                with open(distance_file_path_back, 'r') as distance_file_back:
                    distance_text_back = distance_file_back.read()

                # Read text from Backpoints.txt
                landmark_file_path_back = "outputs/text_files/Backpoints.txt"
                with open(landmark_file_path_back, 'r') as landmark_file_back:
                    landmark_text_back = landmark_file_back.read()

                text = '''
                  <html>
                      <body>
                          <h1>Final Report</h1>
                      </body>
                  </html>
                  '''

                html = "<html><body>"
                html += "<h1>Front Distances</h1>"
                html += "<img src='output_front.png'/><br>"
                html += "<table border='1'>\n"
                html += "<tr><th>Landmarks</th><th>Distance in cm</th></tr>\n"
                # for distance, landmark in front_rows:
                #     html += "<tr>"
                #     html += "<td>{}</td>".format(landmark)
                #     html += "<td>{}</td>".format(distance)
                #     html += "</tr>\n"
                html += "<br>" + distance_text + "<br>"
                html += "<br>" + landmark_text + "<br>"
                html += "</table>"

                html += "<h1>Back Distances</h1>"
                html += "<img src='output_back.png'/><br>"
                html += "<table border='1'>\n"
                html += "<tr><th>Landmarks</th><th>Distance in cm</th></tr>\n"

                html += "<br>" + distance_text_back + "<br>"
                html += "<br>" + landmark_text_back + "<br>"
                html += "</table></body></html>"

                with open('results.html', 'w') as html_file:
                    html_file.write(html)

                # Rest of your code
                options = {
                    'page-size': 'A4',
                    'margin-top': '0mm',
                    'margin-right': '0mm',
                    'margin-bottom': '0mm',
                    'margin-left': '0mm',
                }

                # pdfkit.from_file('results.html', 'Final_Report.pdf', options=options)
                os.system("xdg-open 'results.html'")
                os.system("xdg-open 'Final_Report.pdf'")

# app = MainWindow3D()
# app.run()
