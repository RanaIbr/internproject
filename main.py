import ttkbootstrap as tkinter
import ttkbootstrap as tk
from utils.Interface2D import *
from utils.Interface3D import *


class MainWindow(tkinter.Window):
    def __init__(self):
        super().__init__(themename='vapor')
        self.title("GUI 3D HUMAN BODY")
        self.resizable(0, 0)  # Disable window resizing
        self.style.theme_use("darkly")
        self.frame = tk.Frame(self)
        self.frame.pack()

        icon_path = 'resources/images/logo_19.ico'
        self.iconbitmap(icon_path)

        # Saving User Info
        self.user_info_frame = tkinter.LabelFrame(self.frame, text="User Information")
        self.user_info_frame.grid(row=0, column=0, padx=20, pady=10)

        self.first_name_label = tkinter.Label(self.user_info_frame, text="Name")
        self.first_name_label.grid(row=0, column=0)
        self.last_name_label = tkinter.Label(self.user_info_frame, text="Age")
        self.last_name_label.grid(row=0, column=1)

        self.Tname = tkinter.Entry(self.user_info_frame)
        self.Tage = tkinter.Entry(self.user_info_frame)
        self.Tage.bind("<Key>", self.validate_float)
        self.Tname.grid(row=1, column=0)
        self.Tage.grid(row=1, column=1)

        self.title_label = tkinter.Label(self.user_info_frame, text="Tall")
        self.Ttall = tkinter.Entry(self.user_info_frame)
        self.Ttall.bind("<Key>", self.validate_float)
        self.title_label.grid(row=0, column=2)
        self.Ttall.grid(row=1, column=2)

        self.age_label = tkinter.Label(self.user_info_frame, text="Weight")
        self.Tweight = tkinter.Entry(self.user_info_frame)
        self.Tweight.bind("<Key>", self.validate_float)
        self.age_label.grid(row=2, column=0)
        self.Tweight.grid(row=3, column=0)

        for widget in self.user_info_frame.winfo_children():
            widget.grid_configure(padx=10, pady=5)

        # Saving Course Info
        self.courses_frame = tkinter.LabelFrame(self.frame, text="Gender")
        self.courses_frame.grid(row=1, column=0, sticky="news", padx=20, pady=10)

        self.gender_var = tk.IntVar()
        self.gender_var.set(1)  # Set the initial value to 1 (Male)

        reg_status_var = tkinter.StringVar(value="Not Registered")
        self.Rmale = tkinter.Radiobutton(self.courses_frame, text="Male    ",
                                         variable=self.gender_var, value=1)

        self.Rfemale = tkinter.Radiobutton(self.courses_frame, text="Female",
                                           variable=self.gender_var, value=2)

        self.Rmale.grid(row=0, column=0)
        self.Rfemale.grid(row=1, column=0)

        for widget in self.courses_frame.winfo_children():
            widget.grid_configure(padx=0, pady=0)

        # Accept terms
        self.terms_frame = tkinter.LabelFrame(self.frame, text="Dimension")
        self.terms_frame.grid(row=2, column=0, sticky="news", padx=20, pady=10)

        self.dim_var = tk.IntVar()
        self.dim_var.set(5)  # Set the initial value to 5 (2D)

        accept_var = tkinter.StringVar(value="2D")
        self.check2D = tkinter.Radiobutton(self.terms_frame, text="2D",
                                           variable=self.dim_var, value=5, command=self.toggle_button)

        self.check3D = tkinter.Radiobutton(self.terms_frame, text="3D",
                                           variable=self.dim_var, value=6, command=self.toggle_button2)

        self.check2D.grid(row=0, column=0)
        self.check3D.grid(row=1, column=0)

        for widget in self.user_info_frame.winfo_children():
            widget.grid_configure(padx=10, pady=5)

        # hidden

        # Accept terms
        self.terms_frame_models = tkinter.LabelFrame(self.frame, text="2D Models")
        self.terms_frame_models.grid(row=3, column=0, sticky="new", padx=20, pady=10)

        self.dim_var2 = tk.IntVar()
        self.dim_var2.set(7)  # Set the initial value to 5 (2D)

        accept_var = tkinter.StringVar(value="landmarks")
        self.check2DLandmarks = tkinter.Radiobutton(self.terms_frame_models, text="landmarks        ",
                                                    variable=self.dim_var2, value=7)

        self.check2DFacial = tkinter.Radiobutton(self.terms_frame_models, text="facial                ",
                                                 variable=self.dim_var2, value=8)
        self.check3DMesh = tkinter.Radiobutton(self.terms_frame_models, text="3d mesh facial ",
                                                 variable=self.dim_var2, value=9)

        self.check2DLandmarks.grid(row=0, column=0)
        self.check2DFacial.grid(row=1, column=0)
        self.check3DMesh.grid(row=2, column=0)

        # Button
        self.button = tkinter.Button(self.frame, text="Start", command=self.start)
        self.button.grid(row=4, column=0, sticky="news", padx=20, pady=10)

    def toggle_button(self):
        self.terms_frame_models.grid()

    def toggle_button2(self):
        self.terms_frame_models.grid_remove()

    def validate_float(self, event):
        """Validate the input to accept only float numbers"""
        if event.char.isdigit() or event.char == "." or event.char == "BS" or event.keysym == "BackSpace":
            # Allow digits and the dot character
            pass
        else:
            # Block any other characters
            return "break"

    def start(self):
        self.save_data("outputs/text_files/data.txt")

    def get_data(self):
        model = "none"
        if self.dim_var.get() == 5:
            dimension = "2D"
            if self.dim_var2.get() == 7:
                model = "landmarks"
            if self.dim_var2.get() == 8:
                model = "facial"

        elif self.dim_var.get() == 6:
            dimension = "3D"

        if self.gender_var.get() == 1:
            gender = "male"
        elif self.gender_var.get() == 2:
            gender = "female"
        data = {
            "name": self.Tname.get().strip(),
            "gender": gender,
            "age": self.Tage.get().strip(),
            "tall": self.Ttall.get().strip(),
            "weight": self.Tweight.get().strip(),
            "dimension": dimension,
            "model": model
        }

        return data

    def save_data(self, filename):
        data = self.get_data()
        if self.Tname.get().strip() == "" or self.Ttall.get().strip() == "" or self.Tweight.get().strip() == "" or self.Tage.get().strip() == "":
            messagebox.showinfo("Empty Fields!", "Please check if there are any empty fields.")
        else:
            with open(filename, "w") as f:
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
            if self.dim_var.get() == 5:
                if self.dim_var2.get() == 7:
                    self.open_camera()
                else:
                    self.open_camera_facial()
            elif self.dim_var.get() == 6:
                self.open_3dwindow()

    def open_new_window(self):
        new_window = tk.Toplevel(self.root)
        new_window.title("New Window")

        label = tk.Label(new_window, text="This is a new window!")
        label.pack(padx=10, pady=10)

        button = tk.Button(new_window, text="Close", font=('Arial', 14), command=new_window.destroy)
        button.pack(padx=10, pady=10)

    def open_camera(self):
        camera_window = CameraWindow(self)
        camera_window.grab_set()

    def open_camera_facial(self):
        camera_window_facial = CameraFacialWindow(self)
        camera_window_facial.grab_set()

    def open_3dwindow(self):
        self.withdraw()  # Hide the main window

        window3d = MainWindow3D(self)
        window3d.grab_set()


if __name__ == "__main__":
    main_window = MainWindow()
    main_window.mainloop()
