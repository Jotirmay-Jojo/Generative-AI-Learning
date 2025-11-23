from langchain_community.document_loaders import WebBaseLoader
url='https://en.wikipedia.org/wiki/Black_hole'

loader = WebBaseLoader(url)

docs= loader.load()

print(docs[0].page_content)