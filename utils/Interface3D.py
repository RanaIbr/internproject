import base64
from tkinter import *
from tkinter import filedialog
import customtkinter
from PIL import ImageTk, Image
from utils import *
from captureFront2d import *
from alldistances import *
from finalResult import *
from captureBack2d import *
import time
from automaticLandmarks import *
from loadAndBodyShape import *
from meshFace3d import *
from geometryCroppingAndBoundingBox import *
from bodyLandmarks3d import *
from facialMarks3d import *
from tkinter import ttk
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
from ttkbootstrap import Style
from functions import *
from ttkbootstrap.widgets import Button as ttkButton  # Correct import for ttkbootstrap Button


class MainWindow3D(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent
        style = Style("darkly")  # Apply a theme from ttkbootstrap
        self.title("3D Full Body Human Viewer")
        icon_path = '../resources/images/favicon.ico'
        self.iconbitmap(icon_path)
        # Get the screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(
            f"{screen_width}x{screen_height}+0+0")  # Adjusts the window to top-left corner and matches screen size
        self.create_buttons(style)
        self.path = ""
        self.file_path = ""
        self.path_stl = ""
        self.file_path_stl = ""
        self.html_content = ""
        self.html_label = ""

    def create_buttons(self, style):
        menubar = Menu(self)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Select File", command=self.showDialog)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

        self.current_file = None
        self.Clean3DPLY = None
        if os.path.isfile('../outputs/3d_models/clean.ply'):
            self.Clean3DPLY = '../outputs/3d_models/clean.ply'

        main_frame = Frame(self)
        main_frame.pack(side="top", pady=10, anchor="w")

        giud_text = ("Demo for manual geometry cropping \n"
                     "1) Press 'Y' twice to align geometry with negative direction of y-axis \n"
                     "2) Press 'K' to lock screen and to switch to selection mode \n"
                     "3) Drag for rectangle selection, \n"
                     "or use ctrl + left click for polygon selection \n"
                     "4) Press 'C' to get a selected geometry and to save it \n"
                     "5) Press 'F' to switch to freeview mode")

        self.guide = Label(main_frame, text=giud_text)
        self.guide.pack(side="right", padx=10, pady=10)

        image_path = "../resources/images/trans.png"
        image = Image.open(image_path)
        image = image.resize((400, 500))
        self.photo = ImageTk.PhotoImage(image)
        self.image_label = Label(main_frame, image=self.photo)
        self.image_label.pack(side="right", padx=10, pady=10)

        image_path_ = "../resources/images/kf.png"
        image_ = Image.open(image_path_)
        image_ = image_.resize((270, 130))
        self.photo_ = ImageTk.PhotoImage(image_)
        self.image_label_ = Label(main_frame, image=self.photo_)
        self.image_label_.pack(side="top", padx=10, pady=10)
        self.image_label_.pack(anchor="center")

        self.lable = Label(main_frame, text="Select a .ply file to Start !", wraplength=300)
        self.lable.pack(side="top", padx=10, pady=10)

        top_frame = Frame(main_frame)
        top_frame.pack(side="left", padx=10, pady=10, anchor="w")

        # Labels
        preprocessing_label_frame = LabelFrame(top_frame, text="3D Preprocessing phase")
        preprocessing_label_frame.grid(row=0, column=0, columnspan=2, pady=(10, 0))

        processing_label_frame = LabelFrame(top_frame, text="3D Processing facial phase")
        processing_label_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))

        body_processing_label_frame = LabelFrame(top_frame, text="3D Body Processing")
        body_processing_label_frame.grid(row=8, column=0, columnspan=2, pady=(10, 0))

        other_label_frame = LabelFrame(top_frame, text="Other")
        other_label_frame.grid(row=12, column=0, columnspan=2, pady=(10, 0))

        buttons = [
            ("clean and body shape", lambda: cleanAndBodyShape(self.file_path)),
            ("Geometry cropping",lambda: geometryCropping(self.file_path)),
            ("3d face extraction",lambda: faceExtraction3d(self.file_path)),
            ("3d face mesh landmarks", lambda: faceMeshLandmarks3d(self.file_path)),
            ("left face region", lambda: leftFaceRegion(self.file_path)),
            ("right face region", lambda: rightFaceRegion(self.file_path)),
            ("3d landmarks", lambda: faceMeshLandmarks3d(self.file_path)),
            ("2d landmarks", lambda: landmarks2d(self.file_path)),
            ("Open STL", self.openSTL),
            ("Convert to STL", self.openSTL),
            ("Final Result", self.FinalReportSpot),
            ("Exit", self.parent.destroy)

        ]

        button_positions = [
            preprocessing_label_frame, preprocessing_label_frame,  # Load and body shape, Geometry cropping
            processing_label_frame, processing_label_frame,  # 3d face extraction, left face region
            processing_label_frame, processing_label_frame,  # right face region, 3d face mesh landmarks
            body_processing_label_frame, body_processing_label_frame,  # 3d landmarks, 2d landmarks
            other_label_frame, other_label_frame,  # Open STL, Convert to STL
            other_label_frame, other_label_frame,  # Final Result, Exit
        ]

        for idx, ((text, command), label_frame) in enumerate(zip(buttons, button_positions)):
            button = self.create_button(label_frame, text, command)
            button.grid(row=idx // 2, column=idx % 2, padx=5, pady=5)

        self.label_logo = Label(top_frame)
        self.label_logo.grid(row=13, column=0, padx=10, pady=5)

        self.text_widget = Text(main_frame, width=40, height=30)
        self.text_widget.pack(side="right", padx=10, pady=10)

    def create_label(self, parent, text):
        label = Label(parent, text=text, font=("Helvetica", 12, "bold"))  # Adjust font size here
        return label

    def create_button(self, parent, text, command):
        button_normal_size = (200, 40)  # Adjust button height here
        button_hover_size = (200, 40)  # Adjust button height here
        button_normal_text = text
        button_hover_text = text

        def on_click():
            command()

        def on_enter(event):
            button.configure(
                width=button_hover_size[0],
                height=button_hover_size[1],
                fg_color="#3CB371"
            )

        def on_leave(event):
            button.configure(
                width=button_normal_size[0],
                height=button_normal_size[1],
                fg_color="#2E8B57"
            )

        button = customtkinter.CTkButton(
            master=parent,
            text=button_normal_text,
            command=on_click,
            fg_color="#2E8B57",
            hover_color="#3CB371",
            text_color="#FFFFFF",
            corner_radius=10,
            width=button_normal_size[0],
            height=button_normal_size[1],
            font=("Helvetica", 14),
        )

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

        return button

    def facialMarks(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file")
        else:
            facialMarks3d(self.file_path)

    def loadBodyShape(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file")
        else:
            loadAndShape(self.file_path)

    def geometryCroppingBoundingBox(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file")
        else:
            geometryCroppingAndBoundingBox()

    def finalto(self):
        allDistancesFunction()
        self.insert_text()

    def insert_text(self):
        self.text_widget.config(state="normal")
        self.text_widget.delete('1.0', 'end')
        distance_file_path = "outputs/text_files/Frontdistances.txt"
        with open(distance_file_path, 'r') as distance_file:
            distance_text = distance_file.read()
        distance_file_path_back = "outputs/text_files/Backdistances.txt"
        with open(distance_file_path_back, 'r') as distance_file_back:
            distance_text_back = distance_file_back.read()
        self.text_widget.insert('1.0', distance_text + "\n\n" + distance_text_back)
        self.text_widget.config(state="disabled")

    def showDialog(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*")])
        if self.file_path:
            self.current_file = self.file_path
            self.lable.config(text="Selected file: " + self.file_path)
            self.path = self.file_path

    # Add the rest of your functions here, similar to the ones already modified

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

    def FinalReportSpot(self):
        self.insert_text()
        ImportReportExtra()

# Create and run the main application window
# root = tk.Tk()
# app = MainWindow3D(root)
# root.mainloop()
