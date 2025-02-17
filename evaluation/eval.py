import os
import numpy as np
import hnswlib
import matplotlib.pyplot as plt
import pandas as pd
from evaluation import utils

# Paths to data
mapped_imgs_dir = "data/mapped_imgs"
random_imgs_dir = "random_imgs"
mapped_meta_path = "mapped_meta.csv"
random_meta_path = "random_meta.csv"

# Load metadata
mapped_meta = pd.read_csv(mapped_meta_path)
random_meta = pd.read_csv(random_meta_path)

# User-specified number of environments to test
test_envs = 1  # Adjust as needed
all_envs = list(set(random_meta["env_id"]))
selected_envs = all_envs[:test_envs]

# Filter data based on selected environments
filtered_mapped_meta = mapped_meta[mapped_meta["env_id"].isin(selected_envs)]
filtered_random_meta = random_meta[random_meta["env_id"].isin(selected_envs)]

# Initialize models
models = ["netvlad", "alexnet", "mobilenet", "cohog"]

# Generate feature vectors for mapped and random images
mapped_features = utils.create_feature_vectors(mapped_imgs_dir, filtered_mapped_meta, models)
random_features = utils.create_feature_vectors(random_imgs_dir, filtered_random_meta, models)

# Initialize and populate HNSW indices
hnsw_indices = utils.create_hnsw(mapped_features, models)

# Evaluate recall@n for k1 to k25
results = {model: {f"k{i}": 0 for i in range(1, 26)} for model in models}
total_queries = len(random_features)

for model in models:
    for img_id, query_vector in random_features[model].items():
        indices, _ = hnsw_indices[model].knn_query(query_vector, k=25)
        retrieved_images = [list(mapped_features[model].keys())[i] for i in indices[0]]
        correct_matches = set(filtered_random_meta[filtered_random_meta["image_id"] == img_id]["correct_image_id"].values)

        for k in range(1, 26):
            if any(retrieved_images[i] in correct_matches for i in range(k)):
                results[model][f"k{k}"] += 1

    # Convert counts to recall values
    for k in range(1, 26):
        results[model][f"k{k}"] /= total_queries

# Plot recall@n
plt.figure(figsize=(10, 6))
for model in models:
    recall_values = [results[model][f"k{k}"] for k in range(1, 26)]
    plt.plot(range(1, 26), recall_values, marker='o', label=model)

plt.xlabel("k")
plt.ylabel("Recall@k")
plt.title("Recall@k for Different Models")
plt.legend()
plt.grid(True)
plt.show()
