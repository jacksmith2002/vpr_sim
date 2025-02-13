def load_feature_vectors(data):
    """Loads and decodes feature vectors from JSON."""
    for img_id, entry in data.items():
        for model, encoded_vector in entry["feature_vectors"].items():
            byte_data = base64.b64decode(encoded_vector)
            data[img_id]["feature_vectors"][model] = np.frombuffer(byte_data, dtype=np.float32)