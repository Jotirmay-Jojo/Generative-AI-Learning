from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('newton.txt')

model=ChatGroq(model='llama-3.1-8b-instant')

parser=StrOutputParser()

docs= loader.load()

prompt= PromptTemplate(
    template='Generate a summary of this text \n {text}',
    input_variables=['text']
)

chain= prompt|model|parser

result=chain.invoke({'text':docs[0].page_content})

print(type(docs))

print(len(docs))

print(docs[0].metadata)

print(result)