import sys
import os
from pathlib import Path
from PIL import Image, ImageTk
import torch
from torchvision import transforms as T, models
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox

# ----------------------------
# Config
MODEL_FILE = "resnet50_machining_defects.pth"
INPUT_SIZE = 256
THRESHOLD = 0.1
# ----------------------------

# Handle PyInstaller path
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)

# Load model architecture and weights
def load_model(path=MODEL_PATH, device="cpu"):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 1)
        )
    model.load_state_dict(torch.load("resnet50_machining_defects.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# Preprocess image
def preprocess(img_path, size=INPUT_SIZE):
    img = Image.open(img_path).convert("RGB")
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tf(img).unsqueeze(0)

# Predict
def predict(model, image_path):
    x = preprocess(image_path)
    with torch.no_grad():
        out = model(x)
        prob = torch.sigmoid(out).item()
        if prob >= THRESHOLD:
            label = "Defected ❌"
            color = "red"
        else:
            label = "GOOD ✅"
            color = "green"
        return label, color, prob


# GUI
class DefectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Defect Detection")

        self.model = load_model()

        self.img_label = tk.Label(root, text="No image selected")
        self.img_label.pack(pady=10)

        self.btn_select = tk.Button(root, text="Select Image", command=self.select_image)
        self.btn_select.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return
        try:
            # show image thumbnail
            img = Image.open(file_path)
            img.thumbnail((800, 150))
            img_tk = ImageTk.PhotoImage(img)
            self.img_label.configure(image=img_tk, text="")
            self.img_label.image = img_tk

            # predict
            label, color, prob = predict(self.model, file_path)
            self.result_label.config(text=f"Result: {label} ", fg=color)
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = DefectApp(root)
    root.mainloop()
