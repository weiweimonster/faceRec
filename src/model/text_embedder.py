from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Optional, Union
from src.util.logger import logger
import os

class TextEmbedder:
    _instance: Optional["TextEmbedder"] = None
    model: Optional[SentenceTransformer] = None

    def __new__(cls) -> 'TextEmbedder':
        # Singleton patter to ensure we don't load the model twice
        if cls._instance is None:
            cls._instance = super(TextEmbedder, cls).__new__(cls)
            cls._instance.model = None
            return cls._instance

    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1. Define Local Path
        local_path = "./models/all-mpnet-base-v2"

        # 2. Check if local model exists
        if os.path.exists(local_path):
            print(f"📂 Loading Local Model from {local_path}...")
            self.model = SentenceTransformer(local_path, device=device)
        else:
            # Fallback to download if missing
            print(f"⚠️ Local model not found. Downloading from Hub...")
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

    def embed(self, text: str) -> Optional[List[float]]:
        if not text:
            logger.error("No text provided to embedding model. Returning None")
            return None

        if not self.model:
            raise RuntimeError("Embedder model not initialized")

        vec: Union[np.ndarray, torch.Tensor] = self.model.encode(text, convert_to_numpy=True)

        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            logger.error("No text provided to embedding model. Returning empty list")
            return []

        if self.model is None:
            raise RuntimeError("Embedder model not initialized")

        vecs: Union[np.ndarray, torch.Tensor] = self.model.encode(texts, convert_to_numpy=True)
        return vecs.tolist()

