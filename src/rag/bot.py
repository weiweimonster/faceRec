import os
import base64
import json
import sqlite3
import PIL.Image
from typing import List, Dict, Any

from openai import OpenAI
from google import genai
from google.genai import types

from src.retrieval.engine import SearchEngine


class PhotoAgent:
    def __init__(self, openai_api_key: str, google_api_key: str, search_engine: SearchEngine):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.google_client = genai.Client(api_key=google_api_key)
        self.image_model_name = "gemini-3-pro-image-preview"
        self.engine = search_engine

        # --- TOOLS ---
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Use this to answer questions about existing photos.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search keywords."}
                        },
                        "required": ["query"],
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate a NEW image. Use 'reference_person' if the user wants a specific person from their album.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The detailed visual description."
                            },
                            "reference_person": {
                                "type": "string",
                                "description": "Name of the person to look up in the database (e.g., 'Ethan') to use as a visual reference. Optional."
                            },
                            "pose": {
                                "type": "string",
                                "enum": ["Front", "Side", "Side-Left", "Side-Right"],
                                "description": "Preferred pose for reference photos. Use 'Side' if generating a profile view."
                            },
                            "emotion": {
                                "type": "string",
                                "enum": ["happy", "neutral", "surprise", "angry"],
                                "description": "Expression to match in reference photos."
                            },
                            "shot_type": {
                                "type": "string",
                                "enum": ["Close-up", "Medium-Shot", "Full-Body"],
                                "description": "Zoom level for reference photos."
                            }
                        },
                        "required": ["prompt"],
                    },
                }
            }
        ]

    # --- HELPER: Load Image for Google SDK ---
    def _load_google_image(self, path: str):
        """Loads a local image and converts it for the Google GenAI SDK."""
        try:
            # We use PIL to load, then convert to the specific 'types.Part' or just pass PIL Image
            # The new SDK often accepts PIL images directly in the contents list.
            img = PIL.Image.open(path)
            return img
        except Exception as e:
            print(f"Failed to load reference image {path}: {e}")
            return None

    # --- HELPER: Crop Face from DB ---
    def _get_person_crop(self, image_path: str, person_name: str):
        """
        Loads the image, finds the bounding box for 'person_name' in the DB,
        and returns a cropped PIL Image of just their face (with padding).
        """
        try:
            # 1. Connect to DB to find the bbox
            # We access the SQL path from the engine instance
            conn = sqlite3.connect(self.engine.sql_path)
            cursor = conn.cursor()

            # Find Person ID
            cursor.execute("SELECT person_id FROM people WHERE name LIKE ? LIMIT 1", (person_name,))
            row = cursor.fetchone()
            if not row:
                return None
            person_id = row[0]

            # Find Bounding Box for this specific photo and person
            # We need to match the path. DB might store relative, input might be absolute.
            # We try strict match first, then filename match if needed.
            cursor.execute("""
                           SELECT pf.bounding_box
                           FROM photo_faces pf
                                    JOIN photos ph ON pf.photo_id = ph.photo_id
                           WHERE pf.person_id = ?
                             AND (ph.display_path = ? OR ph.display_path = ?) LIMIT 1
                           """, (person_id, image_path, os.path.basename(image_path)))

            bbox_row = cursor.fetchone()
            conn.close()

            if not bbox_row:
                print(f"⚠️ No bbox found for {person_name} in {image_path}")
                return None

            # 2. Parse BBox
            bbox = json.loads(bbox_row[0])  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(b) for b in bbox]

            # 3. Open Image & Crop
            img = PIL.Image.open(image_path)
            width, height = img.size

            # Add 40% Padding so we get the hair/chin/neck (better for generation)
            pad_x = int((x2 - x1) * 0.4)
            pad_y = int((y2 - y1) * 0.4)

            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(width, x2 + pad_x)
            crop_y2 = min(height, y2 + pad_y)

            cropped_face = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            return cropped_face

        except Exception as e:
            print(f"Error cropping face: {e}")
            return None

    # --- TOOL 1: MEMORY (Unchanged) ---
    def _tool_search_memory(self, query: str) -> Dict[str, Any]:
        # (This remains exactly the same as your previous code)
        print(f"🕵️ Agent decided to SEARCH: {query}")
        image_paths = self.engine.search(query, limit=5)

        if not image_paths:
            return {"answer": "No photos found.", "sources": []}

        content_payload = [{"type": "text", "text": f"User asked: '{query}'. Answer based on these photos."}]
        valid_paths = []
        for path in image_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                content_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
                valid_paths.append(path)

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content_payload}],
                max_tokens=300
            )
            return {"answer": response.choices[0].message.content, "sources": valid_paths}
        except Exception as e:
            return {"answer": f"Error: {e}", "sources": []}

    # --- TOOL 2: ARTIST (Updated with References) ---
    def _tool_generate_image(self, prompt: str, reference_person: str = None, **kwargs) -> Dict[str, Any]:
        print(f"🎨 Agent requested generation. Prompt: '{prompt}', Reference: '{reference_person}'")

        # 1. Extract Filters from kwargs (ignore prompt/person which are explicit)
        filters = {k: v for k, v in kwargs.items() if v is not None}
        if filters:
            print(f"Using Reference Filters: {filters}")

        contents_for_model = [prompt]
        debug_paths = [] # <--- Store paths to show in UI

        # Create debug folder if not exists
        if not os.path.exists("debug_crops"):
            os.makedirs("debug_crops")

        # 2. Look up Reference Photos (if requested)
        if reference_person:
            print(f"Searching for reference photos of: {reference_person}")
            # We search specifically for that person's name
            ref_paths = self.engine.search(
                text_query=reference_person,
                filters=filters,
                limit=6
            )

            faces_found = 0
            if ref_paths:
                print(f"Found {len(ref_paths)} reference images.")
                for i, path in enumerate(ref_paths):

                    # 1. ALWAYS add the Face Crop (Identity is Priority #1)
                    face_crop = self._get_person_crop(path, reference_person)
                    if face_crop:
                        contents_for_model.append(face_crop)
                        faces_found += 1

                        # DEBUG: Save the crop to disk
                        debug_filename = f"debug_crops/crop_{reference_person}_{i}.png"
                        face_crop.save(debug_filename)
                        debug_paths.append(debug_filename)

                    # 2. FOR THE FIRST PHOTO ONLY: Add the Full Image (Body Context)
                    # We only do this once to minimize background noise, but give 1 reference for height/build.
                    if i == 0:
                        try:
                            full_img = PIL.Image.open(path)
                            # Resize if huge to save bandwidth/tokens
                            full_img.thumbnail((1024, 1024))
                            contents_for_model.append(full_img)
                            print("Attached 1 Full Body reference for context.")
                        except Exception:
                            print("Failed to load full image for context. Skipping.")
                            pass
                print(f"Attached {faces_found} face crops.")
            else:
                print("No reference photos found. Proceeding with text only.")

        try:
            # 3. Call Google Gemini 3 Pro
            response = self.google_client.models.generate_content(
                model=self.image_model_name,
                contents=contents_for_model,  # Now contains [Text, Image, Image...]
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"]
                )
            )

            # 4. Save Output
            output_filename = "generated_latest.png"
            image_saved = False

            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        print("Saving generated photos")
                        if part.text is not None:
                            print(part.text)
                        image_data = part.as_image()
                        image_data.save(output_filename)
                        # img_data = base64.b64decode(part.inline_data.data)
                        # with open(output_filename, "wb") as f:
                        #     f.write(img_data)
                        image_saved = True
                        print("Successfully saved generated photos.")
                        break

            if image_saved:
                msg = f"I created this image for '{prompt}'."
                if reference_person:
                    msg += f" I used {faces_found} face shots + 1 body shot of {reference_person}."
                return {
                    "answer": msg,
                    "generated_image": output_filename
                }
            else:
                return {"answer": "The model refused to generate the image (likely safety filters).", "sources": []}

        except Exception as e:
            return {"answer": f"Generation failed: {str(e)}", "sources": []}

    # --- THE BRAIN (Router) ---
    def run(self, user_input: str) -> Dict[str, Any]:
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are a photo agent. If user wants to CREATE an image of a specific person they know, pass that person's name to the 'reference_person' argument."},
                    {"role": "user", "content": user_input}
                ],
                tools=self.tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            if message.tool_calls:
                tool_call = message.tool_calls[0]
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if fn_name == "search_memory":
                    # OLD: return self._tool_search_memory(args["query"])
                    # NEW: Pass everything found (query + filters)
                    return self._tool_search_memory(**args)

                elif fn_name == "generate_image":
                    # OLD: return self._tool_generate_image(prompt=..., reference_person=...)
                    # NEW: Pass everything found (prompt + person + filters)
                    return self._tool_generate_image(**args)

            return {"answer": message.content or "Please clarify.", "sources": []}

        except Exception as e:
            return {"answer": f"Agent Error: {e}", "sources": []}