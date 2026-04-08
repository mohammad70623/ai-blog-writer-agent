from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

GROQ_MODEL = "llama-3.3-70b-versatile"

IMAGES_DIR = "data/images"
