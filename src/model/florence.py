import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from pathlib import Path
import gc
from src.util.logger import logger
from concurrent.futures import ThreadPoolExecutor

# Project root: go up from src/model/florence.py -> src/model -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class VisionScanner:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.model = None
        self.processor = None
        self.hub_id = "Qwen/Qwen2-VL-2B-Instruct"
        self.local_path = PROJECT_ROOT / "models" / "Qwen2-VL-2B-Instruct"

    def load(self) -> None:
        if self.model is not None: return

        # Check if local path exists and is not empty
        if self.local_path.exists() and any(self.local_path.iterdir()):
            load_path = str(self.local_path)
            logger.info(f"Loading Qwen2-VL from LOCAL: {load_path}...")
        else:
            load_path = self.hub_id
            logger.info(f"Local model not found. Downloading Qwen2-VL from HUB: {load_path}...")

        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                load_path,
                torch_dtype=self.dtype,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(load_path)
            logger.info(f"Qwen2-VL loaded on {self.device}")
        except Exception as e:
            logger.error(f"Load failed: {e}")

    def extract_caption_batch(self, image_paths: list[str], batch_size: int = 16) -> list[str]:
        """
        Processes a list of images in parallel batches.
        """
        if self.model is None: self.load()
        if not image_paths: return []

        results = [""] * len(image_paths)
        valid_indices = []
        valid_images = []
        valid_messages = []

        # 1. Parallel Image Loading (CPU Bound)
        # We use threads to open and resize 16 images at once while GPU is busy
        def load_and_resize(path):
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((1280, 1280)) # The 10x Speed Hack
                return img
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
                return None

        with ThreadPoolExecutor() as executor:
            loaded_images = list(executor.map(load_and_resize, image_paths))

        # 2. Prepare Batch for Qwen
        for idx, img in enumerate(loaded_images):
            if img is None: continue

            valid_indices.append(idx)
            valid_images.append(img)

            # Construct message for this specific image
            valid_messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this image in detail."},
                    ],
                }
            ])

        if not valid_messages:
            logger.error("Error processing messages. Returning empty captions")
            return results

        try:
            # 3. Batch Tokenization
            # apply_chat_template doesn't support list-of-lists natively for batching in this specific way
            # so we flatten the structure slightly or use a loop for text prep (fast on CPU)
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in valid_messages
            ]

            # Process vision info for the whole batch
            image_inputs, video_inputs = process_vision_info(valid_messages)

            # The Processor can handle lists of texts and lists of images
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True, # Critical for batching different aspect ratios
                return_tensors="pt",
            ).to(self.device)

            # 4. Batch Generation (The GPU Heavy Lifting)
            # This runs 16x images in parallel on the 5090
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

            # 5. Batch Decode
            output_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # 6. Map results back to original order
            for i, response in enumerate(output_texts):
                clean_text = response
                if "assistant" in response:
                    clean_text = response.split("assistant")[-1].strip()

                original_idx = valid_indices[i]
                results[original_idx] = clean_text

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")

        return results

    def extract_caption(self, image_path: str) -> str:
        if self.model is None: self.load()
        if self.model is None: return ""

        try:
            # Qwen uses a standard chat format
            image = Image.open(image_path).convert("RGB")

            # --- SPEED OPTIMIZATION ---
            # Resize image so the longest edge is max 1280px.
            # This massively reduces the visual tokens (from ~12k to ~1.2k)
            # while keeping enough detail for general captions.
            image.thumbnail((1280, 1280))
            # --------------------------

            # 2. Qwen Chat Format
            messages = [
                {
                    "role": "user",
                    "content": [
                        # Pass the RESIZED PIL object directly, not the path
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this image in detail."},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Generate
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

            # Decode
            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Parse response (strip the prompt)
            response = output_text[0]
            if "assistant" in response:
                return response.split("assistant")[-1].strip()
            return response

        except Exception as e:
            logger.error(f"Qwen error on {image_path}: {e}")
            return ""

    def unload(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
        gc.collect()
        torch.cuda.empty_cache()