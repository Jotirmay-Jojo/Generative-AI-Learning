from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

#app OpenAi ka eek object banaoge
llm = OpenAI(model='gpt-4o-mini')

result=llm.invoke("What is the Captital of India")

print(result)