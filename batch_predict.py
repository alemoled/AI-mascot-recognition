import os
from scripts.compare_embeddings import predict_image

def batch_predict(test_dir, embedding_dir, threshold=0.6):
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("No images found in test directory.")
        return

    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        match, score = predict_image(image_path, embedding_dir, threshold)
        print(f"{image_file} → Match: {match} (Score: {score:.3f})")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    test_dir = os.path.join(base_dir, "test")
    embedding_dir = os.path.join(base_dir, "embeddings")

    print(f"Running predictions on all images in {test_dir}")
    batch_predict(test_dir, embedding_dir, threshold=0.6)
