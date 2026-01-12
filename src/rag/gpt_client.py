from openai import OpenAI
import json
from src.util.logger import logger
from src.util.search_config import SearchFilters


class GPTClient:
    def __init__(self, open_api_key: str):
        self.openai_client = OpenAI(api_key=open_api_key)
        self.model = "gpt-4o"
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search for photos in the database. Uses 'Relevance' ranking by default (finds the specific moment).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                # TODO: Make GPT responded with multiple question for broader extraction. Might also incorporate pass history of the conversation for searching
                                "type": "string",
                                "description": (
                                    "Optimized visual keywords for vector search. "
                                    "CRITICAL: Replace specific names with their visual subject class "
                                    "(e.g., 'Jacob' -> 'person', 'Sarah' -> 'person'). "
                                    "Focus on the action, scene, and lighting. "
                                    "Example Input: 'Jacob under a tree' -> Output: 'person under a tree'."
                                )
                            },
                            "people": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "EXTRACT names mentioned in the prompt. Examples: "
                                               "Input 'Jacob at the beach' -> people=['Jacob']. "
                                               "Input 'Find a photo of jacob side face' -> people=['Jacob']."
                            },
                            "pose": {
                                "type": "string",
                                # Updated to match your Tagger's full capabilities
                                "enum": ["Front", "Side-Left", "Side-Left", "Up", "Down", "Up-Left", "Up-Right", "Down-Left", "Down-Right"],
                                "description": "Head position."
                            },
                            "shot_type": {
                                "type": "string",
                                "enum": ["Close-up", "Medium-Shot", "Full-Body"]
                            },
                            "year": {
                                "type": "integer"
                            },
                            "month": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 12,
                                "description": "The month of the year as a number (1 for January, 12 for December)."
                            },
                            "time_period": {
                                "type": "string",
                                "enum": ["morning", "afternoon", "evening", "night"],
                                "description": "General time of day mentioned or implied (e.g. 'sunset' implies evening)."
                            },
                            "requires_people": {  # Boolean flag
                                "type": "boolean",
                                "description": "Set to TRUE if the user is explicitly looking for specific people, faces, or body poses. Set to FALSE for general scenes, objects, or vibes (e.g. 'sunset', 'nail art', 'beach')."
                            },
                        },
                        "required": ["search_query", "requires_people"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate a new image. Automatically uses 'Strict Quality' mode to find the best high-res reference photos.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Detailed prompt for the image generator."
                            },
                            "search_query": {
                                "type": "string",
                                "description": "Optimized visual keywords for retrieving reference photos. Remove verbs like 'generate', 'create'. Example: 'Jacob walking in New York street'."
                            },
                            "people": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "EXTRACT names mentioned in the prompt. Examples: "
                                               "Input 'Jacob at the beach' -> people=['Jacob']. "
                                               "Input 'Find a photo of jacob side face' -> people=['Jacob']."
                            },
                            # Updated to match Search
                            "pose": {
                                "type": "string",
                                # Updated to match your Tagger's full capabilities
                                "enum": ["Front", "Side-Left", "Side-Left", "Up", "Down", "Up-Left", "Up-Right", "Down-Left", "Down-Right"],
                                "description": "Head position."
                            },
                            "shot_type": {
                                "type": "string",
                                "enum": ["Close-up", "Medium-Shot", "Full-Body"]
                            },
                            "year": {
                                "type": "integer"
                            },
                            "month": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 12,
                                "description": "The month of the year as a number (1 for January, 12 for December)."
                            },
                            "time_period": {
                                "type": "string",
                                "enum": ["morning", "afternoon", "evening", "night"],
                                "description": "General time of day mentioned or implied (e.g. 'sunset' implies evening)."
                            },
                        },
                        "required": ["prompt, search_query", "people"]
                    }
                }
            }
        ]

    def _get_agent_response(self, user_input: str):
        response = None
        try:
            logger.info(f"GPT: {user_input}")
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a photo agent. If user wants to CREATE an image of a specific person they know, pass that person's name to the 'reference_person' argument."},
                    {"role": "user", "content": user_input}
                ],
                tools=self.tools_schema,
                tool_choice="auto"
            )

            if not response:
                logger.error(f"No response from {self.model}")
                return None, "No response from GPT"
            if response.choices and response.choices[0] and response.choices[0].message:
                message = response.choices[0].message
                tool_call = message.tool_calls[0]
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                search_filter = SearchFilters.from_openai_args(fn_name, **args)
                return search_filter, "Successfully load arguments"

            logger.error(f"No message in {self.model}'s response: {response.choices}")
            return None, response.choices[0].message.content or "Please clarify"
        except:
            logger.error(f"Response: {response.choices[0].message}")

