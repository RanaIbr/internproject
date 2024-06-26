import base64
import json
import matplotlib.pyplot as plt
from tkinter import messagebox
import requests
import os
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk
import mediapipe as mp

model_facial = YOLO('../resources/models/best.pt')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

video_path = 0


class CameraFacialWindow(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent
        self.title("Camera Interface")

        main_frame = tk.Frame(self)
        main_frame.pack(side="right", padx=10, pady=10)

        top_frame = tk.Frame(self)
        top_frame.pack(side="top", padx=10, pady=10, anchor="w")

        self.cap = cv2.VideoCapture(video_path)

        self.canvas = tk.Canvas(main_frame, width=640, height=480)
        self.canvas.pack(side="top", padx=10, pady=10)

        self.capture_button = tk.Button(top_frame, width=25, text="Capture", command=self.capture_image)
        self.capture_button.pack(side="top", padx=10, pady=10)

        self.exit_button = tk.Button(top_frame, width=25, text="Exit", command=self.cancel_window)
        self.exit_button.pack(side="top", padx=10, pady=10)

        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model_facial(rgb_frame, save=True)
            annotated_frame = results[0].plot()
            image = Image.fromarray(annotated_frame)
            image = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
            self.canvas.image = image
        self.after(5, self.update)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            results = model_facial(frame, save=True)
            annotated_frame = results[0].plot()
            cv2.imwrite("outputs/images/captured_image.jpg", annotated_frame)
            image = Image.open("outputs/images/captured_image.jpg")
            self.show_confirm_image(image)

    def show_confirm_image(self, image):
        self.new_window = tk.Toplevel(self)
        self.new_window.title("Confirm Image")

        main_frame = tk.Frame(self.new_window)
        main_frame.pack(side="right", padx=10, pady=10)

        top_frame = tk.Frame(self.new_window)
        top_frame.pack(side="top", padx=10, pady=10, anchor="w")

        tk.Label(main_frame, text="Do you want to use this image?").pack(side="top", padx=10, pady=10)
        photo = ImageTk.PhotoImage(image)
        tk.Label(main_frame, image=photo).pack(side="top", padx=10, pady=10)

        upload_btn = tk.Button(top_frame, text="Upload To Server", command=self.save_image, width=25)
        upload_btn.pack(side="top", padx=10, pady=10)

        delete_btn = tk.Button(top_frame, text="Delete", command=self.delete_image, width=25)
        delete_btn.pack(side="top", padx=10, pady=10)

        self.new_window.protocol("WM_DELETE_WINDOW", self.cancel_window)
        self.new_window.mainloop()

    def save_image(self):
        self.new_window.destroy()
        # Implement the image saving and upload logic here
        # Ensure it matches the style and functionality of MainWindow3D

    def delete_image(self):
        os.remove("outputs/images/captured_image.jpg")
        self.new_window.destroy()

    def cancel_window(self):
        self.new_window.destroy()


# Example usage:
# root = tk.Tk()
# app = CameraFacialWindow(root)
# root.mainloop()


class CameraWindow(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent
        self.title("Camera Interface")

        main_frame = tk.Frame(self)
        main_frame.pack(side="right", padx=10, pady=10)

        top_frame = tk.Frame(self)
        top_frame.pack(side="top", padx=10, pady=10, anchor="w")

        # Open the camera
        self.cap = cv2.VideoCapture(0)

        # Create a canvas to display the camera frames
        self.canvas = tk.Canvas(main_frame, width=640, height=480)
        self.canvas.pack(side="top", padx=10, pady=10)

        # Create a button to capture the image
        self.capture_button = tk.Button(top_frame, width=25, text="Capture", command=self.capture)
        self.capture_button.pack(side="top", padx=10, pady=10)

        self.exit_button = tk.Button(top_frame, width=25, text="Exit", command=self.exit)
        self.exit_button.pack(side="top", padx=10, pady=10)

        self.pose_model = mp_pose.Pose()

        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Set the landmark color to red (R, G, B)
                landmark_color = (0, 0, 255)
                mp_drawing.draw_landmarks(
                    rgb_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=landmark_color, thickness=2)
                )

            # Convert the frame to an ImageTk object
            image = Image.fromarray(rgb_frame)
            image = ImageTk.PhotoImage(image)

            # Update the canvas with the new image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
            self.canvas.image = image

        self.after(5, self.update)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            # Save the captured image
            cv2.imwrite("captured_image.jpg", frame)
            messagebox.showinfo("Image Captured", "The image has been saved as captured_image.jpg")

    def capture(self):
        ret, frame = self.cap.read()
        if ret:

            results = pose.process(frame)

            if results.pose_landmarks:
                # Set the landmark color to red (R, G, B)
                landmark_color = (0, 0, 255)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=landmark_color, thickness=2)
                )

            cv2.imwrite("outputs/images/test.jpg", frame)
            image = Image.open("outputs/images/test.jpg")

            self.new_window = tk.Toplevel(self)
            self.new_window.title("Confirm image")

            main_frame = tk.Frame(self.new_window)
            main_frame.pack(side="right", padx=10, pady=10)

            top_frame = tk.Frame(self.new_window)
            top_frame.pack(side="top", padx=10, pady=10, anchor="w")

            tk.Label(main_frame, text="Do you want to use this image?").pack(side="top", padx=10, pady=10)
            photo = ImageTk.PhotoImage(image)
            tk.Label(main_frame, image=photo).pack(side="top", padx=10, pady=10)
            upload_btn = tk.Button(top_frame, text="Upload To server", command=self.save_image, width=25)
            upload_btn.pack(side="top", padx=10, pady=10)

            delete_btn = tk.Button(top_frame, text="Delete", command=self.delete_image, width=25)
            delete_btn.pack(side="top", padx=10, pady=10)
            self.new_window.protocol("WM_DELETE_WINDOW", self.cancel_window)
            self.new_window.mainloop()

    def detectPose(image, pose, display=True):
        output_image = image.copy()

        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(imageRGB)

        height, width, _ = image.shape

        landmarks = []

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image=output_image,
                                      landmark_list=results.pose_landmarks,
                                      connections=mp_pose.POSE_CONNECTIONS)

        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))

        if display:

            plt.figure(figsize=[22, 22])
            plt.imshow(output_image[:, :, ::-1])
            plt.title("Output Image")
            plt.axis('off')
            mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        else:
            return landmarks, output_image

    def save_image(self):
        self.new_window.destroy()
        # model.predict("test.jpg", save=True, imgsz=320, conf=0.5)
        image = cv2.imread('outputs/images/test.jpg')

        height, width, _ = image.shape

        image = cv2.imread('outputs/images/test.jpg')

        display = False
        output_image = image.copy()

        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(imageRGB)

        height, width, _ = image.shape

        landmarks = []
        filename = "data.txt"  # Replace with the actual file name

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

        if display:
            plt.imshow(output_image[:, :, ::-1])
            plt.title("Output Image")
            plt.axis('off')
            mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract the values to variables
        name = data.get("name")
        gender = data.get("gender")
        age = int(data.get("age"))
        tall = int(data.get("tall"))
        weight = int(data.get("weight"))
        dimension = data.get("dimension")
        model = data.get("model")

        # Read the image file as binary data
        with open('outputs/images/test.jpg', 'rb') as file:
            image_data = file.read()

        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Prepare the payload data as a dictionary
        with open('payload.txt', "r") as file:
            file_content = file.read()

        payload = {
            'name': name,
            'gender': gender,
            'age': age,
            'tall': tall,
            'weight': weight,
            'dimension': dimension,
            'model': model,
            'landmarks': file_content
        }
        print(payload)

        self.upload_to_server(payload)
        # else:
        #     return landmarks, output_image

        # self.process_image()

    def upload_to_server(self, payload):

        with open('Final Report.pdf', 'rb') as file:
            files = {
                'pdf': file.read()
            }

            # Read the image file as binary data
        with open('outputs/images/test.jpg', 'rb') as file:
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

    def delete_image(self):
        os.remove("outputs/images/test.jpg")
        self.new_window.destroy()

    def cancel_window(self):
        self.new_window.destroy()

    def exit(self):
        self.cap.release()
        self.destroy()
