import os
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image


# Load pre-trained models
def load_model(model_name):
    if model_name == "alexnet":
        return models.alexnet(pretrained=True).features.eval()
    elif model_name == "mobilenet":
        return models.mobilenet_v2(pretrained=True).features.eval()
    elif model_name == "netvlad":
        raise NotImplementedError("NetVLAD requires a custom implementation or pre-trained model.")
    elif model_name == "cohog":
        raise NotImplementedError("CoHOG requires a custom implementation.")
    else:
        raise ValueError(f"Model {model_name} not supported.")

# Preprocess image for model input
def preprocess_image(image):
    """
    image: NumPy array read by OpenCV in BGR order.
    1) Convert BGR -> RGB
    2) Convert NumPy array to PIL image
    3) Apply torchvision transforms
    4) Return a batch dimension
    """
    # Convert OpenCV's BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert NumPy array (H x W x C) to PIL Image
    pil_image = Image.fromarray(image)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure model-compatible size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Return a batch dimension (1, C, H, W)
    return transform(pil_image).unsqueeze(0)

# Function to split an omnidirectional image into 6 perspective images
def split_omni_image_equirectangular(image):
    """
    Splits a 5376x2688 equirectangular image into 6 sub-images
    laid out as 3 columns (width slices) by 2 rows (height slices).
    Each slice is 1792 wide x 1344 tall.
    """
    h, w, _ = image.shape  # should be 2688, 5376
    face_width = w // 3    # 1792
    face_height = h // 2   # 1344

    # Define the six [x1, y1, x2, y2] crops:
    crops = [
        # Row 0, Col 0
        (0, 0, face_width, face_height),
        # Row 0, Col 1
        (face_width, 0, face_width * 2, face_height),
        # Row 0, Col 2
        (face_width * 2, 0, face_width * 3, face_height),

        # Row 1, Col 0
        (0, face_height, face_width, face_height * 2),
        # Row 1, Col 1
        (face_width, face_height, face_width * 2, face_height * 2),
        # Row 1, Col 2
        (face_width * 2, face_height, face_width * 3, face_height * 2),
    ]

    # Extract each sub-image
    sub_images = []
    for i, (x1, y1, x2, y2) in enumerate(crops):
        # OpenCV slicing: [rows(y), cols(x)]
        crop = image[y1:y2, x1:x2]
        sub_images.append(crop)
    return sub_images

# Generate feature vectors for all images in a folder
def create_feature_vectors(image_dir, metadata, model_names):
    features = {model: {} for model in model_names}
    model_instances = {model: load_model(model) for model in model_names if model != "netvlad" and model != "cohog"}

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing images"):
        image_id = row["image_id"]
        image_path = os.path.join(image_dir, f"{image_id}.jpg")  # Assuming images are in .jpg format
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for PyTorch processing

        # Split into 6 perspective images
        print("Original image shape:", image.shape)
        perspective_images = split_omni_image_equirectangular(image)
        for i, pimg in enumerate(perspective_images):
            print(f"Perspective {i} shape:", pimg.shape)

        # Extract features for each model
        for model in model_names:
            if model in model_instances:
                model_instance = model_instances[model]
                feature_vectors = []

                for perspective_image in perspective_images:
                    image_tensor = preprocess_image(perspective_image)  # Preprocess for model input
                    with torch.no_grad():
                        feature_vector = model_instance(image_tensor)
                        feature_vector = feature_vector.flatten().numpy()
                        feature_vectors.append(feature_vector)

                # Apply pooling strategy
                feature_vectors = np.array(feature_vectors)

                # Use max pooling (Uncomment to switch)
                final_feature_vector = np.max(feature_vectors, axis=0)

                # Use mean pooling (Comment the above and uncomment below to switch)
                # final_feature_vector = np.mean(feature_vectors, axis=0)

                features[model][image_id] = final_feature_vector
            else:
                raise NotImplementedError(f"Feature extraction for {model} is not yet implemented.")

    return features


def create_feature_vectors_no_meta(image_dir, model_names):
    """
    Extract feature vectors for each image in `image_dir` using the specified `model_names`.
    It splits each omnidirectional image into multiple perspectives, extracts features,
    and aggregates them (pooling).
    
    Args:
        image_dir (str): Path to the directory containing images.
        model_names (list): List of model names to use for feature extraction.
        
    Returns:
        dict: A nested dictionary of the form:
              {
                  model_name: {
                      image_id: feature_vector (numpy array),
                      ...
                  },
                  ...
              }
    """
    # Initialize the feature dictionary
    features = {model: {} for model in model_names}

    # Load models except for those not yet implemented
    model_instances = {
        model: load_model(model)
        for model in model_names 
        if model not in ["netvlad", "cohog"]  # Adjust based on your implementation
    }

    # Gather all images from the directory
    # (Adjust extensions as needed: jpg, png, jpeg, etc.)
    image_files = [
        f for f in os.listdir(image_dir) 
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    # Iterate over each image file in the directory
    for img_file in tqdm(image_files, desc="Processing images"):
        # Derive a simple image_id from the filename (without extension)
        image_id = os.path.splitext(img_file)[0]
        image_path = os.path.join(image_dir, img_file)

        # Load image with OpenCV (BGR), then convert to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split into multiple perspective images
        perspective_images = split_omni_image_equirectangular(image)

        # Extract features for each specified model
        for model in model_names:
            if model in model_instances:
                model_instance = model_instances[model]
                feature_vectors = []

                # Extract feature vector for each perspective
                for perspective_image in perspective_images:
                    image_tensor = preprocess_image(perspective_image)
                    with torch.no_grad():
                        feature_vector = model_instance(image_tensor)
                        feature_vector = feature_vector.flatten().numpy()
                        feature_vectors.append(feature_vector)

                # Convert list of arrays to a single numpy array
                feature_vectors = np.array(feature_vectors)

                # ----- Pooling -----
                # Max pooling
                final_feature_vector = np.max(feature_vectors, axis=0)

                # Alternatively, Mean pooling (uncomment to use instead):
                # final_feature_vector = np.mean(feature_vectors, axis=0)
                
                # Store the final pooled feature vector
                features[model][image_id] = final_feature_vector
            else:
                # Handle models that are not implemented or loaded
                raise NotImplementedError(f"Feature extraction for {model} is not implemented.")

    return features
