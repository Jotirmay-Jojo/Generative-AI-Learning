from langchain_core.prompts import ChatPromptTemplate


chat_template=ChatPromptTemplate([
    ('system','You are an expert in {domain}'),
    ('human','explain in breif about the {topic}')
])

prompt=chat_template.invoke({'domain':'AI','topic':'attention in ai'})

print(prompt)