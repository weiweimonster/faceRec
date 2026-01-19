from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Optional, Union, Literal
from pathlib import Path
from src.util.logger import logger
import os

# Project root: go up from src/model/text_embedder.py -> src/model -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class TextEmbedder:
    _instance: Optional["TextEmbedder"] = None
    model: Optional[SentenceTransformer] = None

    def __new__(cls) -> 'TextEmbedder':
        # Singleton pattern to ensure we don't load the model twice
        if cls._instance is None:
            cls._instance = super(TextEmbedder, cls).__new__(cls)
            cls._instance.model = None
        return cls._instance

    def __init__(self) -> None:
        if self.model is not None:
            return  # Already initialized (singleton)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # CHANGED: Point to the new E5 model folder
        local_path = PROJECT_ROOT / "models" / "multilingual-e5-large"
        hub_id = "intfloat/multilingual-e5-large"

        if local_path.exists():
            print(f"📂 Loading Local Model from {local_path}...")
            self.model = SentenceTransformer(str(local_path), device=device)
        else:
            print(f"⚠️ Local model not found at {local_path}. Downloading from Hub...")
            self.model = SentenceTransformer(hub_id, device=device)

        # IMPORTANT: E5 supports up to 512 tokens (vs 384 for MPNet)
        self.model.max_seq_length = 512

    def _prepare_text(self, text: str, input_type: str) -> str:
        """Helper to attach the correct E5 prefix."""
        if input_type == "query":
            return f"query: {text}"
        elif input_type == "passage":
            return f"passage: {text}"
        else:
            # Fallback for safety, though passage is usually the safe default
            logger.warning(f"Unknown input_type '{input_type}', defaulting to 'passage: ' prefix")
            return f"passage: {text}"

    def embed(self, text: str, input_type: Literal["passage", "query"] = "passage") -> Optional[List[float]]:
        """
        Embed a single string.

        Args:
            text: The text to embed.
            input_type: 'passage' for storing in DB (captions), 'query' for search terms.
        """
        if not text:
            logger.error("No text provided to embedding model. Returning None")
            return None

        if not self.model:
            raise RuntimeError("Embedder model not initialized")

        # Prepend the prefix (Crucial for E5)
        formatted_text = self._prepare_text(text, input_type)

        # normalize_embeddings=True is recommended for E5 to enable cosine similarity
        vec: Union[np.ndarray, torch.Tensor] = self.model.encode(
            formatted_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return vec.tolist()

    def embed_batch(self, texts: List[str], input_type: Literal["passage", "query"] = "passage") -> List[List[float]]:
        """
        Embed a list of strings efficiently.
        """
        if not texts:
            logger.error("No text provided to embedding model. Returning empty list")
            return []

        if self.model is None:
            raise RuntimeError("Embedder model not initialized")

        # Apply prefix to all texts in the batch
        formatted_texts = [self._prepare_text(t, input_type) for t in texts]

        vecs: Union[np.ndarray, torch.Tensor] = self.model.encode(
            formatted_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32 # Adjust based on VRAM, 32 is safe for E5-Large
        )
        return vecs.tolist()