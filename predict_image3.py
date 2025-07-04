import os
from scripts.compare_embeddings import predict_image_top3  # ✅ Use the top3 version

def batch_predict(test_dir, embedding_dir, threshold=0.6):
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("No images found in test directory.")
        return

    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        top_matches = predict_image_top3(image_path, embedding_dir, threshold)
        
        # Format output nicely
        result_str = ", ".join([f"{name} ({score:.3f})" for name, score in top_matches])
        print(f"{image_file} → Top matches: {result_str}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    test_dir = os.path.join(base_dir, "test")
    embedding_dir = os.path.join(base_dir, "embeddings")

    print(f"Running predictions on all images in {test_dir}")
    batch_predict(test_dir, embedding_dir, threshold=0.6)
