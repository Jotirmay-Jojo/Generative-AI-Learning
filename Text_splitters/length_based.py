from langchain_text_splitters import TextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('Introduction to Agents.pdf')

docs=loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    separator=''
)

result = splitter.split_documents(docs)
print(result[5].page_content)

# print(docs[6].page_content)