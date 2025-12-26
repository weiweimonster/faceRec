from __future__ import annotations
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# Configuration
DB_PATH = ".db/sqlite/photos.db"
OUTPUT_FILE = "clustering_map.png"

def load_labeled_faces():
    """
    Fetches embeddings and their assigned cluster IDs (person_id).
    Only fetches faces that have been clustered (person_id ~= -1)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # We need the vector AND the label assigned by your previous step
    query = """
        SELECT person_id, embedding_blob
        FROM photo_faces
        WHERE person_id != -1 \
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None, None

    labels = []
    embeddings = []
    print(f"1️⃣ Loading {len(rows)} clustered faces...")
    for pid, blob in rows:
        labels.append(pid)
        embeddings.append(np.frombuffer(blob, dtype=np.float32))
    return np.array(labels), np.array(embeddings)

def visualize_clusters():
    labels, X = load_labeled_faces()
    if X is None:
        print("⚠️ No clustered faces found. Run 'main.py --mode cluster' first.")
        return

    # Run t-SNE (Dimensionality reduction)
    # This crushes 512 dimensions down to 2
    # TODO: need to adjust perplexity according to data points. ROT: perplexity < number_of_samples / 3
    print("2️⃣ Running t-SNE (this might take a moment)...")
    tsne = TSNE(n_components=2, perplexity=35, max_iter=1000, random_state=42)
    X_embedded = tsne.fit_transform(X)

    # Plot
    print("3️⃣ Generating Plot...")
    plt.figure(figsize=(12, 8))

    # Use seaborn for a cleaner scatterplot with automatic legend
    unique_labels = len(set(labels))
    palette = sns.color_palette("hsv", unique_labels)

    sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=labels,
        palette=palette,
        legend="full",
        s=60, # Dot size
        alpha=0.7 # Transparency
    )

    plt.title(f"Face Clustering Visualization ({len(labels)} faces)", fontsize=16)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Person ID")
    plt.tight_layout()

    # 4. Save
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"✅ Visualization saved to {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    visualize_clusters()

