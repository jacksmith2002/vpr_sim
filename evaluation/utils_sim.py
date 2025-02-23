import os
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import hnswlib

# If you have the NetVLAD folder:
from net_vlad.netvlad import NetVLAD

# Import the Hog_descriptor from hog_modified
# Make sure hog_modified.py is in "cohog/hog_modified.py"
from cohog.hog_modified import Hog_descriptor

# ------------------------------------------------------------------------------
# 1) NetVLAD loading code (unchanged)
# ------------------------------------------------------------------------------
def load_netvlad():
    """Loads NetVLAD with VGG16 backbone."""
    checkpoint_path = "net_vlad/vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar"

    # Load VGG16 backbone
    encoder = models.vgg16(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    encoder.load_state_dict(checkpoint["state_dict"], strict=False)
    encoder = torch.nn.Sequential(*list(encoder.features.children())[:-2])  # Remove FC layers

    # Load NetVLAD
    net_vlad = NetVLAD(num_clusters=64, dim=512)
    net_vlad.load_state_dict(checkpoint["state_dict"], strict=False)

    encoder.eval()
    net_vlad.eval()

    return {"encoder": encoder, "netvlad": net_vlad}


# ------------------------------------------------------------------------------
# 2) Model selection function, including "cohog"
# ------------------------------------------------------------------------------
def load_model(model_name):
    """Loads and returns the feature extractor model."""
    if model_name == "alexnet":
        return models.alexnet(pretrained=True).features.eval()
    elif model_name == "mobilenet":
        return models.mobilenet_v3_large(pretrained=True).features.eval()
    elif model_name == "netvlad":
        return load_netvlad()
    elif model_name == "cohog":
        # We just return a string or something indicating "cohog" - 
        # The actual extraction is done in create_feature_vectors().
        return "cohog"
    else:
        raise ValueError(f"Model {model_name} not supported.")


# ------------------------------------------------------------------------------
# 3) Standard PyTorch image preprocessing
# ------------------------------------------------------------------------------
def preprocess_image(image, target_size):
    """Applies PyTorch transforms to an image."""
    preprocess = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess(Image.fromarray(image))


def crop_height(image, goal):
    """Crops top and bottom pixels to make height `goal`."""
    h, w, _ = image.shape
    x_crop = (h - goal) // 2
    return image[x_crop:h - x_crop, :]


def sliding_window(image, window_size=227, stride=113):
    """Generates overlapping patches of size `window_size` with 50% overlap."""
    h, w, _ = image.shape
    return [image[:, x : x + window_size] for x in range(0, w - window_size + 1, stride)]


# ------------------------------------------------------------------------------
# 4) Create feature vectors, adding a CoHOG branch
# ------------------------------------------------------------------------------
def create_feature_vectors(image_dir, metadata, model_name, model_instance, pbar):
    """
    Extracts feature vectors for all images, supporting:
      - NetVLAD
      - CNN models (AlexNet, MobileNet, etc.)
      - CoHOG (via hog_modified)
    """
    features = {}

    # If using NetVLAD, we separate out the encoder and netvlad
    encoder = None
    netvlad = None
    if model_name == "netvlad":
        encoder = model_instance["encoder"]
        netvlad = model_instance["netvlad"]

    for _, row in metadata.iterrows():
        image_id = row["image_idx"]
        image_path = os.path.join(image_dir, f"{image_id}")

        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Warning: Image {image_path} not readable. Skipping.")
            pbar.update(1)
            continue

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 4a) NetVLAD branch
        if model_name == "netvlad":
            image_tensor = preprocess_image(image, (480, 640)).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                vgg_features = encoder(image_tensor)
                vlad_descriptor = netvlad(vgg_features).cpu().numpy()  # Convert to NumPy
            features[image_id] = vlad_descriptor.flatten()

        # 4b) CoHOG branch
        elif model_name == "cohog":
            # Use hog_modified.Hog_descriptor
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hog = Hog_descriptor(img_gray , cell_size=16, bin_size=8)
            hog_block_list = hog.extract()  # This returns a list of block vectors

            # Convert to array, then flatten for a single descriptor
            hog_block_array = np.array(hog_block_list, dtype=np.float32)
            hog_feature_vector = hog_block_array.flatten()

            features[image_id] = hog_feature_vector

        # 4c) CNN models (AlexNet, MobileNet, etc.)
        else:
            # Crop to 224 px tall, do sliding window, run the model, then max-pool
            image = crop_height(image, 224)
            patches = sliding_window(image)
            image_tensors = torch.stack([preprocess_image(patch, 224) for patch in patches])

            # Batch process with the CNN
            with torch.no_grad():
                feature_vectors = model_instance(image_tensors)
                feature_vectors = torch.nn.functional.adaptive_avg_pool2d(feature_vectors, (1, 1))
                feature_vectors = feature_vectors.view(feature_vectors.shape[0], -1).numpy()

            # Max pooling across patches -> final descriptor
            final_feature_vector = np.max(feature_vectors, axis=0)
            features[image_id] = final_feature_vector

        pbar.update(1)

    return features


# ------------------------------------------------------------------------------
# 5) Create HNSW Index (unchanged)
# ------------------------------------------------------------------------------
def create_hnsw(mapped_features, models, space="l2", ef_construction=200, M=16):
    """Creates and populates an HNSW index for each model."""
    hnsw_indices = {}

    for model in models:
        print(f"\n🔄 Initializing HNSW for {model}...")

        if model not in mapped_features:
            print(f"⚠️ No features found for model {model}. Skipping.")
            continue

        image_ids = list(mapped_features[model].keys())
        feature_vectors = np.array([mapped_features[model][img_id] for img_id in image_ids])

        if len(feature_vectors) == 0:
            print(f"⚠️ No valid feature vectors for {model}. Skipping.")
            continue

        dim = feature_vectors.shape[1]
        num_elements = len(image_ids)

        # Initialize HNSW index
        hnsw = hnswlib.Index(space=space, dim=dim)
        hnsw.init_index(max_elements=num_elements,
                        ef_construction=ef_construction,
                        M=M)

        # Convert image names to integer indices
        image_id_map = {img_id: idx for idx, img_id in enumerate(image_ids)}
        int_ids = np.array([image_id_map[img_id] for img_id in image_ids])

        # Add items to the index
        print(f"🔍 Adding {num_elements} vectors to {model} HNSW index...")
        hnsw.add_items(feature_vectors, int_ids)
        hnsw.set_ef(50)

        hnsw_indices[model] = (hnsw, image_id_map)
        print(f"✅ HNSW index created for {model} | Dimension: {dim} | Elements: {num_elements}")

    return hnsw_indices
