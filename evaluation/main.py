import json
import numpy as np
import hnswlib
import matplotlib.pyplot as plt
import base64

# Load the precomputed feature data
with open("place_desc/data/random.json", "r") as f:
    random_data = json.load(f)

with open("place_desc/data/mapped.json", "r") as f:
    mapped_data = json.load(f)

def load_feature_vectors(data):
    """Loads and decodes feature vectors from JSON."""
    for img_id, entry in data.items():
        for model, encoded_vector in entry["feature_vectors"].items():
            byte_data = base64.b64decode(encoded_vector)
            data[img_id]["feature_vectors"][model] = np.frombuffer(byte_data, dtype=np.float32)

# Decode feature vectors
load_feature_vectors(random_data)
load_feature_vectors(mapped_data)

# User-specified number of environments to test
test_envs = 1  # Change this value for different test sizes
all_envs = list(set(entry["environment_id"] for entry in random_data.values()))
selected_envs = all_envs[:test_envs]

# Filter data based on selected environments
filtered_random_data = {k: v for k, v in random_data.items() if v["environment_id"] in selected_envs}
filtered_mapped_data = {k: v for k, v in mapped_data.items() if v["environment_id"] in selected_envs}

# Initialize HNSW indices for each place description model
models = ["netvlad", "alexnet", "mobilenet", "cohog"]
hnsw_indices = {}
image_ids = list(filtered_mapped_data.keys())

dim = len(next(iter(filtered_mapped_data.values()))["feature_vectors"]["netvlad"])

for model in models:
    hnsw_indices[model] = hnswlib.Index(space='cosine', dim=dim)
    hnsw_indices[model].init_index(max_elements=len(filtered_mapped_data), ef_construction=200, M=16)
    to_insert = np.array([filtered_mapped_data[img_id]["feature_vectors"][model] for img_id in image_ids])
    hnsw_indices[model].add_items(to_insert)
    hnsw_indices[model].set_ef(50)

# Evaluate k1 and k4 scores for each model
results = {}
for model in models:
    k1_score = 0
    k4_score = 0
    total_queries = len(filtered_random_data)

    for img_id, entry in filtered_random_data.items():
        query_vector = entry["feature_vectors"][model]
        indices, _ = hnsw_indices[model].knn_query(query_vector, k=4)
        retrieved_images = [image_ids[i] for i in indices[0]]
        correct_matches = entry["nearest_mapped_images"]

        if retrieved_images[0] in correct_matches:
            k1_score += 1
            k4_score += 1
        else:
            for retrieved in retrieved_images:
                if retrieved in correct_matches:
                    k4_score += 1
                    break

    k1_accuracy = k1_score / total_queries
    k4_accuracy = k4_score / total_queries
    results[model] = {"k1": k1_accuracy, "k4": k4_accuracy}
    
    print(f"{model} - k1 accuracy: {k1_accuracy:.4f}, k4 accuracy: {k4_accuracy:.4f}")

# Plot results
plt.figure(figsize=(8, 5))
k1_scores = [results[model]["k1"] for model in models]
k4_scores = [results[model]["k4"] for model in models]

plt.bar(models, k1_scores, label="k1 Accuracy", alpha=0.6)
plt.bar(models, k4_scores, label="k4 Accuracy", alpha=0.6, bottom=k1_scores)
plt.ylabel("Accuracy")
plt.title(f"VPR Evaluation Results Across {test_envs} Envs")
plt.legend()
plt.show()
