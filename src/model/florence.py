import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from pathlib import Path
import gc
from src.util.logger import logger

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