import os
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import hnswlib

#note images will need to be split differently for omni directional ricoh theta

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

        # Preprocess image and generate patches
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = crop_height(image, 224)
        patches = sliding_window(image)

        # Preprocess all patches in batch
        image_tensors = torch.stack([preprocess_image(patch, 224) for patch in patches])

        # Batch process with model
        with torch.no_grad():
            feature_vectors = model_instance(image_tensors)  # Pass entire batch
            feature_vectors = torch.nn.functional.adaptive_avg_pool2d(feature_vectors, (1, 1))
            feature_vectors = feature_vectors.view(feature_vectors.shape[0], -1).numpy()

        if len(feature_vectors) == 0:
            print(f"⚠️ Warning: No features extracted for {image_id}. Skipping.")
            pbar.update(1)
            pbar.refresh()  # ✅ Force refresh
            continue

        # Max pooling across patches
        final_feature_vector = np.max(feature_vectors, axis=0)
        features[image_id] = final_feature_vector

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
