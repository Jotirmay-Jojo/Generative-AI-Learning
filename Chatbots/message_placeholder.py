from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

#chat teplate
chat_template=ChatPromptTemplate([
    ('system','you are a custermer care support agent'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human','{query}')
])
chat_history=[]
#load template
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

#create prompt
result=chat_template.invoke({'chat_history':chat_history,'query':"waht is Virat Kohli Jerrsy number:"})

print(result)