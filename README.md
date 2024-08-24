# Enhanced-Image-Processing-Application

Overview
This project is a comprehensive image processing application that integrates several advanced features, including object detection, face detection, image classification, and various image processing filters. The application leverages machine learning models like MobileNetV2 for classification and YOLOv3 for object detection, along with OpenCV for real-time image processing and GUI creation using Tkinter.

Features
Image Classification: Classify images using the MobileNetV2 model trained on the ImageNet dataset.
Face Detection: Detect faces in images and live camera feeds using Haar cascades.
Object Detection: Identify and highlight objects in images using the YOLOv3 model.
Image Filters: Apply various image processing filters such as blur, sharpen, grayscale, invert, and more.
Live Camera Feed: Capture and process images from a live camera feed.
Image Analysis: Calculate average brightness, generate color histograms, and determine dominant colors in images.
Undo Feature: Revert to previous states of the image with the undo functionality.
Installation
Prerequisites
Python 3.7+
Tkinter
OpenCV
Pillow
TensorFlow
NumPy
Steps
Clone the repository:

bash
Copy code
git clone https://github.com/SarthakKharola/enhanced-image-processing.git
Navigate to the project directory:

bash
Copy code
cd enhanced-image-processing
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Download the pre-trained YOLOv3 weights and configuration files, and place them in the project directory:

YOLOv3 Weights
YOLOv3 Config
COCO Names
Run the application:

bash
Copy code
python main.py
Usage
Open an Image: Use the "Open Image" button to load an image from your file system.
Classify Image: Click the "Classify Image" button to get the class and confidence of the loaded image.
Detect Faces: Use the "Detect Faces" button to highlight faces in the image.
Detect Objects: Click the "Detect Objects" button to detect and label objects using YOLOv3.
Apply Filters: Choose from a variety of filters to process the image.
Live Camera Feed: Enable the live camera feed to process real-time images.
Undo Last Action: Revert the last image processing action.
Project Structure
bash
Copy code
enhanced-image-processing/
├── models/
│   ├── yolov3.weights
│   ├── yolov3.cfg
│   └── coco.names
├── images/
│   ├── example.jpg
│   └── ...
├── src/
│   ├── main.py
│   ├── utils.py
│   └── ...
├── requirements.txt
└── README.md
Future Improvements
Model Training: Integrate options to fine-tune the models with custom datasets.
Real-time Performance: Enhance performance for real-time video processing.
Additional Filters: Expand the list of available image processing filters.
Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

License
This project is licensed under the MIT License. See the LICENSE file for details.
