import os
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import hnswlib
import py360convert
import matplotlib.pyplot as plt

# Note: Images will need to be split differently for omnidirectional Ricoh Theta

def load_model(model_name):
    """Loads and returns the feature extractor model."""
    if model_name == "alexnet":
        return models.alexnet(pretrained=True).features.eval()
    elif model_name == "mobilenet":
        return models.mobilenet_v3_large(pretrained=True).features.eval()
    else:
        raise ValueError(f"Model {model_name} not supported.")

def preprocess_image(image, target_size):
    """Applies PyTorch transforms to an image."""
    preprocess = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess(Image.fromarray(image))

def crop_height(image, goal):
    """Crops top and bottom pixels to make height `goal`."""
    h, w, _ = image.shape
    x_crop = (h - goal) // 2
    return image[x_crop:h-x_crop, :]

def sliding_window(image, window_size=227, stride=113):
    """Generates overlapping patches of size `window_size` with 50% overlap."""
    h, w, _ = image.shape
    return [image[:, x:x + window_size] for x in range(0, w - window_size + 1, stride)]

def convert_to_equirectangular(image):
    """Converts a panoramic image into an equirectangular format first before processing."""
    face_size = 512
    image_resized = cv2.resize(image, (4 * face_size, face_size))
    faces = np.hsplit(image_resized, 4)
    black_face = np.zeros((face_size, face_size, 3), dtype=np.uint8)
    cubemap_faces = [faces[0], faces[1], faces[2], faces[3], black_face, black_face]
    return py360convert.c2e(cubemap_faces, cube_format='list', h=1024, w=2048)

def create_feature_vectors(image_dir, metadata, model_name, model_instance, pbar):
    """Extracts feature vectors for all images using batch processing, with tqdm updates."""
    features = {}

    for _, row in metadata.iterrows():
        image_id = row["image_idx"]
        image_path = os.path.join(image_dir, f"{image_id}")

        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Warning: Image {image_path} not readable. Skipping.")
            pbar.update(1)
            pbar.refresh()  # ✅ Ensure progress bar updates instantly
            continue

        # Convert to equirectangular format first
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = convert_to_equirectangular(image)

        # Process for model compatibility
        h, w, _ = image.shape
        half1 = image[:, :w // 2 + 56]  # Slight overlap
        half2 = image[:, w // 2 - 56:]
        halves = [half1, half2]
        
        feature_vectors = []
        for half in halves:
            half = crop_height(half, 224)
            patches = sliding_window(half)
            image_tensors = torch.stack([preprocess_image(patch, 224) for patch in patches])
            
            with torch.no_grad():
                extracted_features = model_instance(image_tensors)
                extracted_features = torch.nn.functional.adaptive_avg_pool2d(extracted_features, (1, 1))
                extracted_features = extracted_features.view(extracted_features.shape[0], -1).numpy()
            
            if len(extracted_features) > 0:
                feature_vectors.append(np.max(extracted_features, axis=0))

        if feature_vectors:
            features[image_id] = np.max(feature_vectors, axis=0)  # Max pooling across halves
        
        pbar.update(1)  
        pbar.refresh()  # ✅ Ensures the bar updates in real-time

    return features


def create_hnsw(mapped_features, models, space='l2', ef_construction=200, M=16):
    """
    Creates and populates an HNSW index for each model.
    """

    hnsw_indices = {}

    for model in models:
        print(f"\n🔄 Initializing HNSW for {model}...")

        if model not in mapped_features:
            print(f"⚠️ No features found for model {model}. Skipping.")
            continue

        # Get image IDs and feature vectors
        image_ids = list(mapped_features[model].keys())  # Corrected
        feature_vectors = np.array([mapped_features[model][img_id] for img_id in image_ids])  # Corrected

        if len(feature_vectors) == 0:
            print(f"⚠️ No valid feature vectors for {model}. Skipping.")
            continue

        # Get feature vector dimension
        dim = feature_vectors.shape[1]  # No change
        num_elements = len(image_ids)

        # Initialize HNSW index
        hnsw = hnswlib.Index(space=space, dim=dim)
        hnsw.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)

        # Convert image names to integer indices for HNSW
        image_id_map = {img_id: idx for idx, img_id in enumerate(image_ids)}
        int_ids = np.array([image_id_map[img_id] for img_id in image_ids])  # Convert to numeric

        # Add items to the index
        print(f"🔍 Adding {num_elements} vectors to {model} HNSW index...")
        hnsw.add_items(feature_vectors, int_ids)

        # Set query-time parameters
        hnsw.set_ef(50)

        hnsw_indices[model] = (hnsw, image_id_map)  # Save index and mapping
        print(f"✅ HNSW index created for {model} | Dimension: {dim} | Elements: {num_elements}")

    return hnsw_indices