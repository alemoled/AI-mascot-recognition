import os
import csv
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from scripts.compare_embeddings import predict_image_top3  # Ensure this returns top 3 predictions
import numpy as np

def evaluate(test_root, embedding_dir, threshold=0.6, output_csv="evaluation_results.csv"):
    y_true = []
    y_pred = []
    results = []

    for label in os.listdir(test_root):
        folder_path = os.path.join(test_root, label)
        if not os.path.isdir(folder_path):
            continue

        for image_name in os.listdir(folder_path):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(folder_path, image_name)

            top_preds = predict_image_top3(image_path, embedding_dir, threshold=threshold)
            prediction = top_preds[0][0]  # Best match
            confidence = top_preds[0][1]  # Score of best match

            y_true.append(label)
            y_pred.append(prediction)

            results.append({
                "Image": image_name,
                "True Label": label,
                "Predicted Label": prediction,
                "Confidence": round(confidence, 4),
                "Top 3 Predictions": ", ".join([f"{name} ({score:.3f})" for name, score in top_preds])
            })

            print(f"{image_name}: True = {label} | Predicted = {prediction} | Confidence = {confidence:.3f}")

    # Save results to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Image", "True Label", "Predicted Label", "Confidence", "Top 3 Predictions"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nResults saved to {output_csv}")

    # Print metrics
    print("\n--- Evaluation Results ---")
    labels = sorted(set(y_true + y_pred))
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred, labels=labels))

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    test_dir = os.path.join(base_dir, "test_set")       # Folder with label subfolders
    embedding_dir = os.path.join(base_dir, "embeddings")  # Folder with .npy embeddings

    evaluate(test_dir, embedding_dir, threshold=0.6)
