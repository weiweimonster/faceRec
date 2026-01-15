import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Optional, Any, Dict
import os
import gc
from src.util.logger import logger

class VisionScanner:
    def __init__(self) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Any = None
        self.processor: Optional[AutoProcessor] = None
        self.local_model_path = "./models/Florence-2-large"

    def load(self) -> None:
        if self.model is None:
            logger.warning("Vision model is already loaded, skipping")
            return

        if os.path.exists(self.local_model_path):
            logger.info(f"📂 Loading Vision Scanner from LOCAL: {self.local_model_path}...")
            load_path = self.local_model_path
        else:
            logger.error(f"Cannot find vision model to load. Exiting...")
            return

        self.processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
                load_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="eager"
            )

        logger.info("Vision Scanner loaded")

    def extract_caption(self, image_path: str) -> str:
        if self.model is None:
            logger.info("Loading Vision Scanner Model")
            self.load()
        if not os.path.exists(image_path):
            logger.error(f"⚠️ Warning: Image not found at {image_path}")
            return ""

        try:
            image: Image.Image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            prompt: str = "<MORE_DETAILED_CAPTION>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.float16)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=3,
                )

            generated_text: str = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            parsed: Dict[str, Any] = self.processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image.width, image.height)
            )

            return str(parsed.get(prompt, ""))

        except Exception as e:
            logger.error(f"❌ Vision Error on {image_path}: {e}")
            return ""

    def unload(self) -> None:
        """
        Free up VRAM when scanning is finished.
        """
        if self.model is None:
            return

        logger.info("🗑️ Unloading Vision Scanner...")
        del self.model
        del self.processor
        self.model = None
        self.processor = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()