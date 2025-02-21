import os
import numpy as np
import utils

# Initialize models
models = ["alexnet", "mobilenet"]

# Directory containing your .JPG images
test_images_dir = 'data/model_testing_images'

# 1) Gather the list of JPG files
test_image_files = [
    f for f in os.listdir(test_images_dir)
    if f.lower().endswith('.jpg')
]

# 2) Generate feature vectors
image_features = utils.create_feature_vectors_no_meta(test_images_dir, models)

# 3) Store similarity results
similarity_results = {model: {} for model in models}

# 4) Pick a query image file (e.g., the first file in the list)
query_image_file = "0_dist.JPG"

print(f"Query Image: {query_image_file}")

# 5) In `create_feature_vectors_no_meta(...)`, you might store features
#    as `features[model][image_id]` where `image_id` is often the *stem*
#    (i.e., file name without extension). If that's the case, we do:
query_image_id = os.path.splitext(query_image_file)[0]

for model in models:
    # Retrieve query vector
    query_vector = image_features[model][query_image_id]
    print(query_vector).shape()

    # Compute similarity with all other images
    for img_file in test_image_files:
        # Get the dictionary key used during feature extraction
        img_id = os.path.splitext(img_file)[0]
        feature_vector = image_features[model][img_id]

        # Cosine similarity
        similarity = np.dot(query_vector, feature_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(feature_vector)
        )
        similarity_results[model][img_id] = similarity

    # Sort images based on similarity score
    sorted_images = sorted(
        similarity_results[model].items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"Similarity scores for model: {model}")
    for img, score in sorted_images:
        print(f"{img}: {score:.4f}")
    print("\n")
