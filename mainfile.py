import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from collections import Counter
import threading

# Initialize the MobileNetV2 model for image classification
def initialize_mobilenet():
    return MobileNetV2(weights="imagenet")

# Initialize the Haar cascade for face detection
def initialize_haar_cascade(path):
    return cv2.CascadeClassifier(path)

# Initialize YOLOv3 for object detection
def initialize_yolo(weights_path, cfg_path, names_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Load models and assets using specified paths from the code repository
model = initialize_mobilenet()
face_cascade = initialize_haar_cascade('b:/A drive/transfer file/projects/image processing/.vscode/project/haarcascade_frontalface_default.xml')
net, classes, output_layers = initialize_yolo(
    'b:/A drive/transfer file/projects/image processing/.vscode/project/yolov3.weights',
    'b:/A drive/transfer file/projects/image processing/.vscode/project/yolov3.cfg',
    'b:/A drive/transfer file/projects/image processing/.vscode/project/coco.names'
)

img = None
original_img = None
processed_img = None
image_history = []  # To keep track of images
history_index = -1  # Keep track of the current image in history

# Function to calculate average brightness
def calculate_brightness(image):
    return np.mean(image)

# Function to generate color histogram
def color_histogram(image):
    histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram

# Function to find dominant colors
def dominant_colors(image, num_colors=3):
    pixels = image.reshape(-1, 3)
    colors = Counter(map(tuple, pixels)).most_common(num_colors)
    return colors

# Function to classify an image using MobileNetV2
def classify_image():
    global img
    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return
    
    # Prepare image for MobileNetV2
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    
    # Predict and display the class
    predictions = model.predict(image)
    label = decode_predictions(predictions, top=1)[0][0]
    class_name = label[1]
    confidence = label[2]
    
    # Additional image information
    width, height = img.shape[1], img.shape[0]
    avg_brightness = calculate_brightness(img)
    histogram = color_histogram(img)
    total_pixels = width * height
    aspect_ratio = width / height
    dominant_colors_info = dominant_colors(img)

    # Formatting the output in Markdown
    result_text = f"### Image Information\n" \
                  f"- **Class**: {class_name}\n" \
                  f"- **Confidence**: {confidence:.2f}\n" \
                  f"- **Dimensions**: {width}x{height}\n" \
                  f"- **Total Pixels**: {total_pixels}\n" \
                  f"- **Aspect Ratio**: {aspect_ratio:.2f}\n" \
                  f"- **Average Brightness**: {avg_brightness:.2f}\n" \
                  f"- **Color Histogram**: {histogram.tolist()}\n" \
                  f"- **Dominant Colors**: {dominant_colors_info}\n"
                  
    update_info_box(result_text)

# Function to detect faces using Haar cascade
def detect_faces():
    global img, processed_img
    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    processed_img = img.copy()
    add_to_history(processed_img)  # Add to history
    show_image(processed_img)
    update_info_box(f"### Detected Faces\n- **Count**: {len(faces)}")

# Function to detect objects using YOLOv3
def detect_objects():
    global img, processed_img
    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return
    
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids, confidences, boxes = [], [], []
    detected_objects = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
            detected_objects.append(f"{label} ({confidence:.2f})")
    
    processed_img = img.copy()
    add_to_history(processed_img)  # Add to history
    show_image(processed_img)
    update_info_box("### Detected Objects\n- " + "\n- ".join(detected_objects))

def add_to_history(image):
    global history_index
    image_history[:] = image_history[:history_index + 1]  # Truncate history if we are adding a new image
    image_history.append(image)
    history_index += 1  # Move to the latest image in history

def undo_last_action():
    global img, processed_img, history_index
    if history_index > 0:
        history_index -= 1
        processed_img = image_history[history_index]
        show_image(processed_img)
        update_info_box("Undid last action.")
    else:
        messagebox.showwarning("Warning", "No action to undo.")

def open_file():
    global img, original_img, processed_img, image_history, history_index
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Could not open image.")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = img.copy()
        processed_img = img.copy()
        add_to_history(processed_img)  # Add to history
        show_image(img)
        update_info_box("Image loaded successfully.")

def reset_image():
    global img, original_img, processed_img
    if original_img is not None:
        img = original_img.copy()
        processed_img = original_img.copy()
        add_to_history(processed_img)  # Add to history
        show_image(img)
        update_info_box("Image reset to original state.")
    else:
        messagebox.showwarning("Warning", "No original image to reset to.")

def save_image():
    global processed_img
    if processed_img is None:
        messagebox.showwarning("Warning", "No image to save.")
        return
    
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
    if save_path:
        img_to_save = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_to_save)
        update_info_box("Image saved successfully.")

def apply_filter(filter_type):
    global img, processed_img
    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return

    if filter_type == 'Blur':
        processed_img = cv2.GaussianBlur(img, (15, 15), 0)
    elif filter_type == 'Sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed_img = cv2.filter2D(img, -1, kernel)
    elif filter_type == 'Grayscale':
        processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif filter_type == 'Invert':
        processed_img = cv2.bitwise_not(img)
    elif filter_type == 'Canny':
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.Canny(gray, 100, 200)
    elif filter_type == 'Rotate':
        processed_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif filter_type == 'Brightness':
        beta = 50  # Brightness adjustment value
        processed_img = cv2.convertScaleAbs(img, beta=beta)
    elif filter_type == 'Shear':
        rows, cols, _ = img.shape
        M = np.float32([[1, 0.5, 0], [0, 1, 0]])  # Shear matrix
        processed_img = cv2.warpAffine(img, M, (cols, rows))
    elif filter_type == 'Translate':
        rows, cols, _ = img.shape
        M = np.float32([[1, 0, 50], [0, 1, 50]])  # Translate by 50 pixels
        processed_img = cv2.warpAffine(img, M, (cols, rows))
    elif filter_type == 'Sepia':
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                  [0.349, 0.686, 0.168],
                                  [0.393, 0.769, 0.189]])
        processed_img = cv2.transform(img, sepia_filter)
        processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)
    elif filter_type == 'Edge Detection':
        processed_img = cv2.Canny(img, 100, 200)

    add_to_history(processed_img)  # Add to history
    show_image(processed_img)  # Ensure the processed image is displayed
    update_info_box(f"{filter_type} filter applied.")

def show_image(image):
    if len(image.shape) == 2: 
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(image)

    # Resize image to fit within the canvas
    image.thumbnail((canvas.winfo_width(), canvas.winfo_height()), Image.LANCZOS)
    image = ImageTk.PhotoImage(image)

    canvas.create_image(0, 0, anchor=tk.NW, image=image)
    canvas.image = image  # Keep a reference to avoid garbage collection

def _on_mouse_wheel(event):
    if info_label.winfo_containing(event.x_root, event.y_root) is info_label:
        info_label.yview_scroll(int(-1*(event.delta/120)), "units")

# Function to update the information box with new text
def update_info_box(info_text):
    info_label.config(state=tk.NORMAL)  # Allow editing
    info_label.delete(1.0, tk.END)  # Clear current content
    info_label.insert(tk.END, info_text)  # Insert new text
    info_label.config(state=tk.DISABLED)  # Disallow editing
    info_label.yview(tk.END)  # Scroll to the end

def start_live_feed():
    global is_live_camera
    is_live_camera = True
    threading.Thread(target=update_live_feed).start()

def update_live_feed():
    global img, processed_img, is_live_camera
    cap = cv2.VideoCapture(0)
    while is_live_camera:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_img = img.copy()
        
        # Detect faces in the frame
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        show_image(img)
        root.update_idletasks()
        root.update()

    cap.release()

def stop_live_feed():
    global is_live_camera
    is_live_camera = False

# GUI setup
root = tk.Tk()
root.title("Enhanced Image Processing Application")
root.geometry("1200x700")
root.configure(bg='#2a2a72')

# Configure style for buttons
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12), padding=15)
style.map('TButton',
          foreground=[('!disabled', 'black')],
          background=[('!disabled', '#00ccff'), ('active', '#0099cc')],
          relief=[('pressed', 'groove'), ('!pressed', 'ridge')])

# Canvas to display the image
canvas = tk.Canvas(root, width=800, height=600, bg='#ffffff', borderwidth=2, relief='sunken')
canvas.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

# Information box to display classification and detection results
info_label = tk.Text(root, bg='#2a2a72', fg='white', font=('Helvetica', 14), wrap=tk.WORD, height=15, width=50)
info_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
info_label.config(state=tk.DISABLED)  # Set it to read-only initially

# Frame for buttons and operations
frame = tk.Frame(root, bg='#2a2a72')
frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

# Scrollable frame for buttons
canvas_frame = tk.Canvas(frame, bg='#2a2a72')
scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas_frame.yview)
scrollable_frame = tk.Frame(canvas_frame, bg='#2a2a72')

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas_frame.configure(
        scrollregion=canvas_frame.bbox("all")
    )
)

canvas_frame.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)
canvas_frame.configure(yscrollcommand=scrollbar.set)

canvas_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Operation buttons with increased width and height
operations = [
    ("Open Image", open_file),
    ("Reset Image", reset_image),
    ("Save Image", save_image),
    ("Undo Last Action", undo_last_action),
    ("Classify Image", classify_image),
    ("Detect Faces", detect_faces),
    ("Detect Objects", detect_objects),
    ("Apply Blur Filter", lambda: apply_filter('Blur')),
    ("Apply Sharpen Filter", lambda: apply_filter('Sharpen')),
    ("Apply Grayscale Filter", lambda: apply_filter('Grayscale')),
    ("Apply Invert Filter", lambda: apply_filter('Invert')),
    ("Apply Canny Edge Detection", lambda: apply_filter('Canny')),
    ("Rotate Image", lambda: apply_filter('Rotate')),
    ("Adjust Brightness", lambda: apply_filter('Brightness')),
    ("Apply Shear", lambda: apply_filter('Shear')),
    ("Apply Translation", lambda: apply_filter('Translate')),
    ("Apply Sepia Filter", lambda: apply_filter('Sepia')),
    ("Apply Edge Detection", lambda: apply_filter('Edge Detection')),
    ("Start Live Feed", start_live_feed),
    ("Stop Live Feed", stop_live_feed)
]

for (text, command) in operations:
    ttk.Button(scrollable_frame, text=text, command=command, width=25).pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

# Status bar
status_bar = tk.Label(root, text="Welcome to the Image Processing Application!", bg='#2a2a72', fg='white', anchor='w')
status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10)

# Adding mouse hover functionality for scrolling
info_label.bind("<Enter>", lambda e: root.bind("<MouseWheel>", _on_mouse_wheel))
info_label.bind("<Leave>", lambda e: root.unbind("<MouseWheel>"))

root.mainloop()
