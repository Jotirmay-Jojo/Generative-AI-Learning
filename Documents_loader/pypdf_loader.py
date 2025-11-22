from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Introduction to Agents.pdf')

docs= loader.load()

print(docs[6].page_content)