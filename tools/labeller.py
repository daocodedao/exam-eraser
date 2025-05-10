import os
import tkinter as tk
from PIL import Image, ImageTk
import random

image_folder = ''
valid_extensions = ('.png', '.jpg', '.jpeg')

labels = {
    'Handwritten': 'Handwritten',
    'Printed': 'Printed',
    'Noise': 'Noise'
}

def is_labeled(filename):
    return any(label in filename for label in set(labels.values()))

image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(valid_extensions) and not is_labeled(f)
])

random.shuffle(image_files)

class LabelTool:
    def __init__(self, master):
        self.master = master
        self.index = 0
        self.total = len(image_files)

        self.label = tk.Label(master, text="", font=("Helvetica", 12, "bold"))
        self.label.pack(pady=5)

        self.image_label = tk.Label(master)
        self.image_label.pack(pady=10)

        self.legend = tk.Label(master, text=self.get_legend_text(), font=("Helvetica", 10))
        self.legend.pack(pady=5)

        self.status = tk.Label(master, text="", fg='green')
        self.status.pack(pady=5)

        master.focus_set()
        master.bind("<KeyPress>", self.handle_key_press)

        self.image_label.bind("<KeyPress>", self.handle_key_press)

        self.load_image()

    def get_legend_text(self):
        return (
            "⬅ Left Shift: Handwritten    ➡ Right Shift: Printed    ␣ Space: Noise"
        )

    def load_image(self):
        if self.index >= self.total:
            self.status.config(text="All images labeled!")
            self.image_label.config(image='')
            self.label.config(text='')
            return

        filename = image_files[self.index]
        self.label.config(text=f"Image {self.index + 1}/{self.total}: {filename}")

        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).resize((200, 200))
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_img)

    def handle_key_press(self, event):
        if event.keysym == "Shift_L":
            label = 'Handwritten'
        elif event.keysym == "Shift_R":
            label = 'Printed'
        elif event.keysym == "space":
            label = 'Noise'
        else:
            return

        self.apply_label(label)

    def apply_label(self, label):
        old_name = image_files[self.index]
        name, ext = os.path.splitext(old_name)
        new_name = f"{name}_{label}{ext}"
        os.rename(
            os.path.join(image_folder, old_name),
            os.path.join(image_folder, new_name)
        )
        print(f"Labeled '{old_name}' as {label}")
        self.index += 1
        self.load_image()

if not image_files:
    print("No unlabeled images found")
else:
    root = tk.Tk()
    root.title("Image Labeler")
    app = LabelTool(root)
    root.mainloop()
