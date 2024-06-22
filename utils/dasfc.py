import customtkinter
# Create the main window
app = customtkinter.CTk()  # Use CTk() instead of Tk()
app.geometry("400x200")
app.title("Expandable Button Example")
# Define the button's normal and hover properties
button_normal_size = (200, 40)
button_hover_size = (220, 50)
button_normal_text = "Click Me"
button_hover_text = "Click Me for More Information"
# Define a function to be called when the button is clicked
def button_click():
    print("Button clicked!")
# Define functions to handle mouse enter and leave events
def on_enter(event):
    button.configure(
        width=button_hover_size[0],
        height=button_hover_size[1],
        text=button_hover_text
    )
def on_leave(event):
    button.configure(
        width=button_normal_size[0],
        height=button_normal_size[1],
        text=button_normal_text
    )
# Create a modern-looking button
button = customtkinter.CTkButton(
    master=app,
    text=button_normal_text,
    command=button_click,
    fg_color="#2E8B57",  # Button color
    hover_color="#3CB371",  # Button color when hovered
    text_color="#FFFFFF",  # Text color
    corner_radius=10,  # Rounded corners
    width=button_normal_size[0],
    height=button_normal_size[1],
)
# Bind the hover events to the button
button.bind("<Enter>", on_enter)
button.bind("<Leave>", on_leave)
# Place the button in the window
button.place(relx=0.5, rely=0.5, anchor="center")
# Start the main loop
app.mainloop()