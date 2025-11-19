from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

chat_history = [
    SystemMessage(content="You are a AI assistant: "),
    HumanMessage(content="What is the full name of Madum Curie")
]

result = model.invoke(chat_history)
chat_history.append(AIMessage(content=result.content))
print(chat_history)