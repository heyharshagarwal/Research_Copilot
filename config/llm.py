from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
if not os.environ["GROQ_API_KEY"]:
    raise ValueError("Please set GROQ_API_KEY in your environment")

llm = ChatGroq(
    model_name="qwen/qwen3-32b",
    temperature=0.1,
    max_tokens=1024
)