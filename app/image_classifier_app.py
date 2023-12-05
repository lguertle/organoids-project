import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox
import shutil
import os

class ImageClassifier(tk.Tk):
    def __init__(self, image_paths, labels):
        super().__init__()
        self.configure(bg='white')
        self.image_paths = image_paths
        self.labels = labels
        self.current_image_index = 0
        self.classifications = {}
        self.selected_buttons = {}  # Track selected labels for each image
        self.selected_label = None
        self.delete_mode = False  # Flag to indicate delete mode
        # Dark theme for buttons
        self.dark_bg = "#C3C3C3"  # Dark gray background

        self.title("Image Classifier")
        self.geometry("800x700")  # Or your preferred size

        self.image_frame = tk.Frame(self, borderwidth=2, relief="solid", bg="black")  # Adjust border width and color as needed
        self.image_frame.pack(padx=10, pady=10)  # Add some padding around the frame
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        self.label_frame = tk.Frame(self)
        self.label_frame.pack()
        self.label_buttons = self.create_label_buttons()

        self.add_label_frame = tk.Frame(self)
        self.add_label_frame.pack()
        self.new_label_entry = tk.Entry(self.add_label_frame)
        self.new_label_entry.pack(side=tk.LEFT)
        self.add_label_button = tk.Button(self.add_label_frame, text="Add Label", command=self.add_new_label, bg=self.dark_bg,font=("Helvetica", 10))
        self.add_label_button.pack(side=tk.LEFT)

        self.navigation_frame = self.create_navigation_buttons()

        self.delete_label_button = tk.Button(self, text="Delete Label", 
                                     command=lambda: self.enable_delete_mode(), state=tk.NORMAL, bg=self.dark_bg, font=("Helvetica", 8))
        self.bind("<Button-3>", self.enable_delete_mode)
        self.delete_label_button.pack(pady=10)

        self.save_button = tk.Button(self, text="Save", command=self.save_images, state=tk.DISABLED,
                             font=("Helvetica", 14), bg="#4492DB", fg="white", 
                             padx=10, pady=5, borderwidth=2, relief="raised")
        self.save_button.pack()
        self.save_button.pack_forget()

        self.update_image()

    def enable_delete_mode(self, event=None):  # Accept an optional event parameter
        if event is not None or self.delete_mode:  # Toggle delete mode off if it's a right-click or if delete mode is already on
            self.delete_mode = False
            self.config(cursor="")  # Restore default cursor
        else:
            self.delete_mode = True
            self.config(cursor="cross")  # Set cursor to crosshair

    def label_button_click(self, label):
        if self.delete_mode:
            self.delete_label(label)
        else:
            self.classify_image(label)

    def delete_label(self, label):
        # Ask for confirmation before deletion
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the label '{label}'?"):
            # Delete the selected label
            self.labels.remove(label)
            self.label_buttons[label].destroy()
            del self.label_buttons[label]
            self.delete_mode = False
            self.config(cursor="")  # Restore default cursor
            self.classifications = {path: lbl for path, lbl in self.classifications.items() if lbl != label}
            self.update_button_states()
            self.check_all_labeled()

    def check_all_labeled(self):
        if len(self.classifications) == len(self.image_paths):
            self.save_button.pack()  # Show the save button
            self.save_button.config(state=tk.NORMAL)
        else:
            self.save_button.pack_forget()  # Hide the save button
            self.save_button.config(state=tk.DISABLED)

    def create_label_buttons(self):
        buttons = {}
        for label in self.labels:
            button = tk.Button(self.label_frame, text=label, 
                            command=lambda l=label: self.label_button_click(l),
                            font=("Helvetica", 12), bg=self.dark_bg, activebackground="#d0d0d0", padx=10, pady=5)
            button.pack(side=tk.LEFT, padx=5, pady=10)
            buttons[label] = button
        return buttons

    def add_new_label(self):
        new_label = self.new_label_entry.get()
        if new_label and new_label not in self.labels:
            self.labels.append(new_label)
            button = tk.Button(self.label_frame, text=new_label, 
                               command=lambda l=new_label: self.label_button_click(l),
                               font=("Helvetica", 12), bg=self.dark_bg, activebackground="#d0d0d0", padx=10, pady=5)
            button.pack(side=tk.LEFT, padx=5, pady=10)
            self.label_buttons[new_label] = button
        self.new_label_entry.delete(0, tk.END)

    def update_image(self):
        image = Image.open(self.image_paths[self.current_image_index])
        image = image.resize((400, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.update_button_states()

    def create_navigation_buttons(self):
        frame = tk.Frame(self)
        frame.pack()
        back_button = tk.Button(frame, text="Back", command=self.previous_image,
                                font=("Helvetica", 14), bg=self.dark_bg, activebackground="#a0a0a0", padx=50, pady=10)
        back_button.pack(side=tk.LEFT, padx=15, pady=10)
        next_button = tk.Button(frame, text="Next", command=self.next_image,
                                font=("Helvetica", 14), bg=self.dark_bg, activebackground="#a0a0a0", padx=50, pady=10)
        next_button.pack(side=tk.LEFT, padx=15, pady=10)
        return frame

    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.update_image()
            self.update_button_states()

    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image()
            self.update_button_states()

    def classify_image(self, label):
        current_image_path = self.image_paths[self.current_image_index]
        if self.selected_buttons.get(self.current_image_index) == label:
            # Unclicking the button - remove the label
            del self.selected_buttons[self.current_image_index]
            if current_image_path in self.classifications:
                del self.classifications[current_image_path]
        else:
            # Clicking the button - add or update the label
            self.selected_buttons[self.current_image_index] = label
            self.classifications[current_image_path] = label

        self.update_button_states()
        self.check_all_labeled()

    def update_button_states(self):
        selected_label = self.selected_buttons.get(self.current_image_index)
        for label, button in self.label_buttons.items():
            if label == selected_label:
                button.config(bg='lightgreen')
            else:
                button.config(bg='#f0f0f0')

    def save_images(self):
        # Ask for confirmation before saving
        if messagebox.askyesno("Confirm Save", "Are you sure you want to save the labels?"):
            # Create 'images_classified' directory if it doesn't exist
            classified_images_dir = os.path.join(script_directory, "images_classified")
            # If 'images_classified' exists and is not empty, clear its contents
            if os.path.exists(classified_images_dir) and os.listdir(classified_images_dir):
                for file_or_dir in os.listdir(classified_images_dir):
                    file_or_dir_path = os.path.join(classified_images_dir, file_or_dir)
                    if os.path.isfile(file_or_dir_path):
                        os.remove(file_or_dir_path)
                    else:  # It's a directory
                        shutil.rmtree(file_or_dir_path)

            # Create 'images_classified' directory if it doesn't exist
            if not os.path.exists(classified_images_dir):
                os.makedirs(classified_images_dir)

            for label in self.labels:
                # Create a directory for each label inside 'images_classified'
                label_dir = os.path.join(classified_images_dir, label)
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)

            for image_path, label in self.classifications.items():
                # Copy the image to the corresponding label directory
                dest_path = os.path.join(classified_images_dir, label, os.path.basename(image_path))
                shutil.copy(image_path, dest_path)  # Use shutil.move to move instead of copy

            messagebox.showinfo("Save Complete", "Images have been saved into label folders.")
        else:
            # User chose not to save
            messagebox.showinfo("Save Cancelled", "The save operation has been cancelled.")

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_folder = os.path.join(script_directory, "images")
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
    labels = ["Label1", "Label2", "Label3"]

    app = ImageClassifier(image_paths, labels)
    app.mainloop()
