import tkinter as tk
from tkinter import ttk

# Define the loadBodyShape function
def loadBodyShape():
    print("Load body shape function called")

# Create the main application window
root = tk.Tk()

# Create a frame in the main window
top_frame = tk.Frame(root)
top_frame.pack()

# Define a custom style for the button
style = ttk.Style()
style.configure("RoundedButton.TButton",
                font=('Helvetica', 12),
                padding=10,
                relief="flat",
                background="#4CAF50",
                foreground="white",
                borderwidth=0,
                focuscolor="#4CAF50")
style.map("RoundedButton.TButton",
          background=[('active', '#45a049')],
          foreground=[('active', 'white')],
          relief=[('pressed', 'flat')],
          focuscolor=[('pressed', '#45a049')])

# Create a new button with the custom style
button_geo_cropping = ttk.Button(top_frame, text="Load and body shape", width=25,
                                 command=loadBodyShape, style="RoundedButton.TButton")
button_geo_cropping.pack(side="top", padx=10, pady=5)

# Add a rounded shape effect by adding a canvas behind the button
canvas = tk.Canvas(top_frame, width=button_geo_cropping.winfo_reqwidth() + 20,
                   height=button_geo_cropping.winfo_reqheight() + 20, bg="#4CAF50", bd=0,
                   highlightthickness=0)
canvas.create_oval(0, 0, button_geo_cropping.winfo_reqwidth() + 20,
                   button_geo_cropping.winfo_reqheight() + 20, fill="#4CAF50", outline="#4CAF50")
canvas.pack(side="top", padx=10, pady=5)
canvas.create_window((button_geo_cropping.winfo_reqwidth() // 2) + 10,
                     (button_geo_cropping.winfo_reqheight() // 2) + 10, window=button_geo_cropping)

# Start the Tkinter main loop
root.mainloop()
