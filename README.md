COCO Dataset Visualization with Flask

This project is a Flask web application designed to visualize images and their annotations from the COCO (Common Objects in Context) dataset. It specifically demonstrates fetching and displaying image captions and instance segmentation masks with bounding boxes directly in your web browser.
Features
 * COCO Annotation Integration: Loads and parses COCO annotations (captions and instances) using pycocotools.
 * Dynamic Image Visualization: Displays COCO images from your local dataset.
 * Caption Overlay: Shows associated captions for selected images.
 * Instance Segmentation Overlay: Renders instance segmentation masks and bounding boxes with category labels on images.
 * Web Interface: A simple Flask application serves up the visualizations.
 * On-the-Fly Plot Generation: Generates and saves visualization plots as PNGs, which are then displayed in the web app.
   
Setup & Running the Application

Follow these steps to get the application running on your local machine.

1. Prerequisites
Make sure you have:
 * Python 3.x
 * pip (Python package installer)
   
2. Get the Code
Clone this repository to your local machine:
git clone <repository_url>
cd <repository_folder>

3. Install Dependencies
Install the necessary Python libraries:
pip install Flask matplotlib numpy Pillow pycocotools

Note: pycocotools can sometimes be tricky to install. If you encounter issues, especially on Windows, consider installing Cython first (pip install Cython) or look for pre-compiled wheels.

4. Download COCO Dataset
You'll need parts of the COCO 2017 dataset. Download the following and extract them into a single parent directory:
 * Images: train2017.zip, val2017.zip
 * Annotations: annotations_trainval2017.zip
You can find these on the official COCO website: cocodataset.org/#download

Recommended Data Structure:
Your extracted COCO data should reside within a main folder, which you'll point to as your DATA_ROOT:

<YOUR_DATA_ROOT>
  
├── train2017/

├── val2017/

└── annotations_trainval2017/

    ├── captions_train2017.json
    
    ├── captions_val2017.json
    
    ├── instances_train2017.json
    
    ├── instances_val2017.json
    
    └── ...

5. Configure DATA_ROOT
Open app.py and update the DATA_ROOT variable to the absolute path of the directory where you've placed the COCO dataset.

# app.py
DATA_ROOT = r'C:\Users\vaish\Downloads\archive (3)' # <-- CHANGE THIS PATH

 * Windows users: Use a raw string (r'...') or forward slashes (/).
 * macOS/Linux users: Use standard forward slashes.
 * 
6. Run the Application
From your project directory in the terminal, execute:
python app.py

Open your web browser and navigate to http://127.0.0.1:5000/. The application will load, process, and display a few random COCO images with their respective annotations.
Project Structure
.
├── app.py                  # Main Flask application and visualization logic


├── static/                 # Directory for static web files


│   └── generated_images/   #  <-- Generated plot images are saved here


└── templates/

    └── index.html          # HTML template for the web interface

Troubleshooting

 * DATA_ROOT Path: Double-check your DATA_ROOT path in app.py. Incorrect paths are a common source of errors.
 * pycocotools Install: If you face installation errors, especially related to C++ compilers, ensure you have the necessary build tools for your OS (e.g., Microsoft Visual C++ Build Tools on Windows).
 * No Images Displayed: Verify that the static/generated_images folder is created and contains PNG files after running app.py. Check your browser's developer console for errors.
