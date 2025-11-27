from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

#step1 your source documents
documents=[
    Document(page_content="Langchain helps developers build LLM applications easily"),
    Document(page_content="Chroma is a vector database optimized for LLM-based search"),
    Document(page_content="Embeddings convert text into high-dimensional vectors"),
    Document(page_content="OpenAI provides poweful embedding models"),
]

#step2 Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#step3 Create Chroma vector store in memory
vectorstore=Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
)

retriver = vectorstore.as_retriever(search_kwargs={'k':2})

query="What is Chroma used for"
results=retriver.invoke(query)

for i , doc in enumerate(results):
    print(f"\n --result{i+1}---")
    print(doc.page_content)
    
# Retriver can be use to create multiple strategies on how to retrieve data    
#---------#--------


#fixed strategy of vector store function(simlarity_search)
result=vectorstore.similarity_search(query,k=2)

for i , doc in enumerate(result):
    print(f"\n --result{i+1}---")
    print(doc.page_content)