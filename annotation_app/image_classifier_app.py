import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox
from tkinter import filedialog
from PIL import UnidentifiedImageError
import shutil
import os

class ImageClassifier(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Image Classifier")
        self.geometry("400x200")  # Set the initial size of the window

        # Create a button to select the image folder
        self.select_folder_button = tk.Button(self, text="Select the folder containing your images",
                                              command=self.select_image_folder)
        self.select_folder_button.pack(pady=50)  # Center the button with some padding

    def select_image_folder(self):
        while True:
            image_folder = filedialog.askdirectory(title="Select Directory with Images")
            if not image_folder:
                break  # User cancelled; you might want to close the app or handle this differently

            if self.check_image_readability(image_folder):
                self.initialize_classifier(image_folder)
                break
            else:
                # Unreadable images found, prompt to choose another folder
                continue

    def initialize_classifier(self, image_folder):
        self.select_folder_button.destroy()
        self.image_folder = image_folder
        self.image_paths = [os.path.join(self.image_folder, img) for img in os.listdir(self.image_folder)]
        self.configure(bg='white')
        self.labels = []
        self.current_image_index = 0
        self.classifications = {}
        self.selected_buttons = {}  # Track selected labels for each image
        self.selected_label = None
        self.delete_mode = False  # Flag to indicate delete mode
        # Dark theme for buttons
        self.dark_bg = "#DCDCDC"  # Dark gray background

        self.title("Image Classifier")
        self.geometry("800x700")  # Or your preferred size

        self.image_frame = tk.Frame(self, borderwidth=2, relief="solid", bg="black")  # Adjust border width and color as needed
        self.image_frame.pack(padx=10, pady=10)  # Add some padding around the frame
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        self.label_frame = tk.Frame(self)
        self.label_frame.pack()
        self.label_buttons = self.create_label_buttons()
        self.bind_keyboard_events()

        self.add_label_frame = tk.Frame(self)
        self.add_label_frame.pack()
        self.new_label_entry = tk.Entry(self.add_label_frame)
        self.new_label_entry.pack(side=tk.LEFT)
        self.add_label_button = tk.Button(self.add_label_frame, text="Add Label", command=self.add_new_label, bg=self.dark_bg,font=("Helvetica", 10))
        self.add_label_button.pack(side=tk.LEFT)

        self.navigation_frame = self.create_navigation_buttons()

        # Label to display current image number over total images
        self.image_count_label = tk.Label(self, text=f"Image 1/{len(self.image_paths)}")
        self.image_count_label.pack()

        self.delete_label_button = tk.Button(self, text="Delete Label", 
                                    command=lambda: self.enable_delete_mode(), state=tk.NORMAL, bg=self.dark_bg, font=("Helvetica", 8))
        self.bind("<Button-3>", self.enable_delete_mode)
        self.delete_label_button.pack(pady=10)

        # Bind arrow keys
        self.bind("<Left>", lambda event: self.previous_image())
        self.bind("<Right>", lambda event: self.next_image())
        self.focus_set()  # Ensure the window has focus

        self.save_button = tk.Button(self, text="Save", command=self.save_images, state=tk.DISABLED,
                            font=("Helvetica", 14), bg="#4492DB", fg="white", 
                            padx=10, pady=5, borderwidth=2, relief="raised")
        self.save_button.pack()
        self.save_button.pack_forget()

        self.update_image()

    def check_image_readability(self, image_folder):
        for img_file in os.listdir(image_folder):
            img_path = os.path.join(image_folder, img_file)
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verifies that an image is readable
            except (UnidentifiedImageError, IOError):
                # Handle unreadable image
                messagebox.showerror("Error", f"The image '{img_file}' in the selected folder cannot be read. Please select an appropriate folder.")
                return False
        return True

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
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the label '{label}'?"):
            self.labels.remove(label)
            self.label_buttons[label].destroy()
            del self.label_buttons[label]
            self.delete_mode = False
            self.config(cursor="")  # Restore default cursor
            self.classifications = {path: lbl for path, lbl in self.classifications.items() if lbl != label}
            self.update_button_states()
            self.check_all_labeled()
            # Update the keyboard bindings
            self.update_keyboard_bindings()

    def check_all_labeled(self):
        if len(self.classifications) == len(self.image_paths):
            self.save_button.pack()  # Show the save button
            self.save_button.config(state=tk.NORMAL)
        else:
            self.save_button.pack_forget()  # Hide the save button
            self.save_button.config(state=tk.DISABLED)

    def create_label_buttons(self):
        self.key_to_label_map = {}
        buttons = {}
        for index, label in enumerate(self.labels):
            button = tk.Button(self.label_frame, text=label, 
                            command=lambda l=label: self.label_button_click(l),
                            font=("Helvetica", 12), bg=self.dark_bg, padx=10, pady=5)
            button.pack(side=tk.LEFT, padx=5, pady=10)
            buttons[label] = button
            if index < 9:  # Limit to first 9 labels
                self.key_to_label_map[str(index + 1)] = label
        return buttons
    
    def bind_keyboard_events(self):
        for key, label in self.key_to_label_map.items():
            self.bind_all(key, lambda event, l=label: self.key_pressed(event, l))

    def add_new_label(self):
        new_label = self.new_label_entry.get().strip()
        if new_label and new_label not in self.labels:
            self.labels.append(new_label)
            button = tk.Button(self.label_frame, text=new_label, 
                            command=lambda l=new_label: self.label_button_click(l),
                            font=("Helvetica", 12), bg=self.dark_bg, padx=10, pady=5)
            button.pack(side=tk.LEFT, padx=5, pady=10)
            self.label_buttons[new_label] = button
            # Update the keyboard bindings
            self.update_keyboard_bindings()
        self.new_label_entry.delete(0, tk.END)
        self.focus_set()

    def update_keyboard_bindings(self):
        for key in list(self.key_to_label_map.keys()):
            self.unbind_all(key)
        self.key_to_label_map = {str(index + 1): label for index, label in enumerate(self.labels) if index < 9}
        for key, label in self.key_to_label_map.items():
            self.bind_all(key, lambda event, l=label: self.key_pressed(event, l))

    def key_pressed(self, event, label):
        if self.focus_get() == self.new_label_entry:
            # Do nothing if the entry field has focus
            return
        self.classify_image(label)

    def update_image(self):
        image = Image.open(self.image_paths[self.current_image_index])
        image = image.resize((400, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.image_count_label.config(text=f"Image {self.current_image_index + 1}/{len(self.image_paths)}")
        self.update_button_states()

    def create_navigation_buttons(self):
        frame = tk.Frame(self)
        frame.pack()
        back_button = tk.Button(frame, text="Back", command=self.previous_image,
                                font=("Helvetica", 14), bg=self.dark_bg, padx=50, pady=10)
        back_button.pack(side=tk.LEFT, padx=15, pady=10)
        next_button = tk.Button(frame, text="Next", command=self.next_image,
                                font=("Helvetica", 14), bg=self.dark_bg, padx=50, pady=10)
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
                button.config(bg=self.dark_bg)

    def save_images(self):
        # Ask for confirmation before saving
        parent_directory = os.path.dirname(self.image_folder)
        if messagebox.askyesno("Confirm Save", "Are you sure you want to save the labels?"):
            # Create 'images_classified' directory if it doesn't exist
            classified_images_dir = os.path.join(parent_directory, "images_classified")
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

            messagebox.showinfo("Save Complete", f"Images have been saved into {classified_images_dir}")
        else:
            # User chose not to save
            messagebox.showinfo("Save Cancelled", "The save operation has been cancelled.")

if __name__ == "__main__":
    app = ImageClassifier()
    app.mainloop()
