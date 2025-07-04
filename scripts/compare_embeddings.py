import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from scripts.model import get_embedding_model
from torch.nn.functional import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same transform as before
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_known_embeddings(embedding_dir):
    embeddings = {}
    for file in os.listdir(embedding_dir):
        if file.endswith(".npy"):
            name = file[:-4]
            vector = np.load(os.path.join(embedding_dir, file))
            embeddings[name] = vector
    return embeddings

def predict_image(image_path, embedding_dir, threshold=0.6):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model = get_embedding_model().to(device)
    with torch.no_grad():
        embedding = model(tensor).squeeze().cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)

    known = load_known_embeddings(embedding_dir)

    best_score = -1
    best_match = "unknown"

    for name, known_emb in known.items():
        score = cosine_similarity(torch.tensor(embedding), torch.tensor(known_emb), dim=0).item()
        if score > best_score:
            best_score = score
            best_match = name

    if best_score < threshold:
        return "unknown", best_score
    return best_match, best_score

def predict_image_top3(image_path, embedding_dir, threshold=0.6):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model = get_embedding_model().to(device)
    with torch.no_grad():
        embedding = model(tensor).squeeze().cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)

    known = load_known_embeddings(embedding_dir)

    input_emb = torch.tensor(embedding).to(device)

    scores = []
    for name, known_emb in known.items():
        known_emb_tensor = torch.tensor(known_emb).to(device)
        score = torch.nn.functional.cosine_similarity(input_emb, known_emb_tensor, dim=0).item()
        scores.append((name, score))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    # Filter out those below threshold
    filtered = [(name, score) for name, score in scores if score >= threshold]

    # If none passed threshold, return unknown
    if not filtered:
        return [("unknown", 0.0)]

    # Return top 3 or fewer
    return filtered[:3]
