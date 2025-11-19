from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model=ChatGroq(model="llama-3.1-8b-instant")

result=model.invoke("What is the capital of india")

print(result.content)