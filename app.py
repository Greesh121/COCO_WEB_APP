import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from flask import Flask, render_template, url_for

# Initialize the Flask application
app = Flask(__name__)

# --- 0. Configuration: Define your data paths ---
# IMPORTANT: This DATA_ROOT path MUST be correct and use forward slashes or a raw string.
# Example using raw string (r''):
DATA_ROOT = r'C:\Users\vaish\Downloads\archive (3)'

# Image directories - USE THE DATA_ROOT VARIABLE
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'train2017')
VAL_IMG_DIR = os.path.join(DATA_ROOT, 'val2017')
TEST_IMG_DIR = os.path.join(DATA_ROOT, 'test2017')

CAPTIONS_TRAIN_JSON = os.path.join(DATA_ROOT, 'annotations_trainval2017', 'captions_train2017.json')
CAPTIONS_VAL_JSON = os.path.join(DATA_ROOT, 'annotations_trainval2017', 'captions_val2017.json')
INSTANCES_TRAIN_JSON = os.path.join(DATA_ROOT, 'annotations_trainval2017', 'instances_train2017.json')
INSTANCES_VAL_JSON = os.path.join(DATA_ROOT, 'annotations_trainval2017', 'instances_val2017.json')

# Path where generated image plots will be saved (within the static folder)
GENERATED_IMG_FOLDER = os.path.join(app.root_path, 'static', 'generated_images')
os.makedirs(GENERATED_IMG_FOLDER, exist_ok=True) # Create the folder if it doesn't exist

# --- 1. Data Loading Utility Functions

def load_coco_annotations(json_path):
    """Loads COCO annotations using pycocotools.COCO API."""
    if not os.path.exists(json_path):
        print(f"Error: Annotation file not found at {json_path}")
        return None
    try:
        coco = COCO(json_path)
        print(f"Loaded COCO annotations from: {json_path}")
        return coco
    except Exception as e:
        print(f"Error loading COCO annotations from {json_path}: {e}")
        print("Please ensure the path is correct and the JSON file is valid COCO format.")
        return None

def load_image(image_dir, file_name):
    """Loads an image using PIL."""
    image_path = os.path.join(image_dir, file_name)
    try:
        img = Image.open(image_path).convert('RGB')
        return np.array(img) # Return as NumPy array for plotting
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}. Please check your image directory and file names.")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# --- 2. New Function to Generate Caption Plots for Web ---
def generate_caption_plots():
    image_data_for_html = [] # Stores paths and captions for HTML
    coco_captions_train = load_coco_annotations(CAPTIONS_TRAIN_JSON)

    if coco_captions_train:
        img_ids = coco_captions_train.getImgIds()
        # Take a few random samples, or adjust as needed
        sample_img_ids = random.sample(img_ids, min(len(img_ids), 3))

        for i, img_id in enumerate(sample_img_ids):
            img_info = coco_captions_train.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            image_np = load_image(TRAIN_IMG_DIR, file_name) # Renamed to image_np for clarity

            if image_np is not None:
                ann_ids = coco_captions_train.getAnnIds(imgIds=img_id)
                annotations = coco_captions_train.loadAnns(ann_ids)

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(image_np) # Use the numpy array here
                ax.set_title(f"Image ID: {img_id} ({file_name})")
                ax.axis('off')

                # Prepare captions for display in HTML
                caption_text = ""
                if annotations:
                    for ann in annotations:
                        caption_text += f"- {ann.get('caption', 'N/A')}\n"
                else:
                    caption_text = "- No captions found for this image.\n"

                # Save the figure to a file
                plot_filename = f"caption_plot_{img_id}.png"
                plot_filepath = os.path.join(GENERATED_IMG_FOLDER, plot_filename)
                plt.savefig(plot_filepath)
                plt.close(fig) # Close the figure to free up memory

                # Store the URL for the HTML template
                image_data_for_html.append({
                    'url': url_for('static', filename=f'generated_images/{plot_filename}'),
                    'captions': caption_text.strip()
                })
    return image_data_for_html

# --- 3. New Function to Generate Instance Segmentation Plots for Web ---
def generate_instance_plots():
    image_urls_for_html = []
    coco_instances_train = load_coco_annotations(INSTANCES_TRAIN_JSON)

    if coco_instances_train:
        img_ids_with_anns = coco_instances_train.getImgIds(catIds=coco_instances_train.getCatIds())
        sample_img_ids_seg = random.sample(img_ids_with_anns, min(len(img_ids_with_anns), 3))

        for i, img_id in enumerate(sample_img_ids_seg):
            img_info = coco_instances_train.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            image_np = load_image(TRAIN_IMG_DIR, file_name)

            if image_np is not None:
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(image_np)
                ax.set_title(f"Image ID: {img_id} ({file_name}) with Instances")
                ax.axis('off')

                ann_ids = coco_instances_train.getAnnIds(imgIds=img_id)
                annotations = coco_instances_train.loadAnns(ann_ids)

                colors = plt.cm.get_cmap('tab10', max(1, len(annotations)))

                for j, ann in enumerate(annotations):
                    category_info = coco_instances_train.loadCats(ann['category_id'])[0]
                    category_name = category_info['name']

                    bbox = ann.get('bbox')
                    rect_color = colors(j)

                    if bbox and len(bbox) == 4:
                        x, y, w, h = bbox
                        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=rect_color, linewidth=2)
                        ax.add_patch(rect)
                        ax.text(x, y - 5, category_name, color='white', fontsize=10,
                                bbox=dict(facecolor=rect_color, alpha=0.7, edgecolor='none', pad=1))

                    if 'segmentation' in ann:
                        segmentation = ann['segmentation']
                        try:
                            rles = maskUtils.frPyObjects(segmentation, img_info['height'], img_info['width'])
                            binary_mask = maskUtils.decode(rles)
                            if binary_mask.ndim == 3:
                                binary_mask = np.sum(binary_mask, axis=2)
                            binary_mask = (binary_mask > 0).astype(np.uint8)

                            mask_rgb = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.float32)
                            mask_rgb[binary_mask > 0] = [*rect_color[:3], 0.4]

                            ax.imshow(mask_rgb)
                        except Exception as e:
                            print(f"  Warning: Could not process segmentation for annotation {ann.get('id')}: {e}")

                # Saving the figure to a file
                plot_filename = f"instance_plot_{img_id}.png"
                plot_filepath = os.path.join(GENERATED_IMG_FOLDER, plot_filename)
                plt.savefig(plot_filepath)
                plt.close(fig) # Close the figure

                # Store the URL for the HTML template
                image_urls_for_html.append(url_for('static', filename=f'generated_images/{plot_filename}'))
    return image_urls_for_html

# --- Flask Routes ---
@app.route('/')
def home():
    # Call the functions to generate and save plots
    caption_image_data = generate_caption_plots()
    instance_image_urls = generate_instance_plots()

    # Pass the generated image URLs and caption data to the HTML template
    return render_template('index.html',
                           caption_images=caption_image_data,
                           instance_images=instance_image_urls)

if __name__ == "__main__":
    app.run(debug=True)
