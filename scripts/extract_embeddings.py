import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from scripts.model import get_embedding_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation pipeline for each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

def extract_embeddings(data_dir, output_dir):
    model = get_embedding_model().to(device)

    os.makedirs(output_dir, exist_ok=True)

    for character in os.listdir(data_dir):
        char_path = os.path.join(data_dir, character)
        if not os.path.isdir(char_path): continue

        embeddings = []

        for img_file in tqdm(os.listdir(char_path), desc=f"Processing {character}"):
            try:
                img_path = os.path.join(char_path, img_file)
                image = Image.open(img_path).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = model(tensor).squeeze().cpu().numpy()
                    embedding = embedding.flatten()  # Shape: (512,)
                    embedding = embedding / np.linalg.norm(embedding)  # Normalize

                embeddings.append(embedding)
            except Exception as e:
                print(f"Error with {img_file}: {e}")

        # Save the average embedding for this fursuiter
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            save_path = os.path.join(output_dir, f"{character}.npy")
            np.save(save_path, avg_embedding)
            print(f"Saved embedding for {character} to {save_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "dataset")
    output_dir = os.path.join(base_dir, "embeddings")

    print(f"Extracting embeddings from {data_dir}")
    extract_embeddings(data_dir, output_dir)
