from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#create / connect to vector store
vector_store = Chroma(
    collection_name='my_collection',
    embedding_function=embeddings,
    persist_directory='./chromadb' #optional for local persistance
)

docs = [
    Document(
        page_content=' Virat kolhi : An accomplished batter and former captain, known for his aggressive style and leading India to numerous victories. ', 
        metadata={"source":"test"}),
    
    Document(
        page_content="""The current captain, a right-handed batsman with a high score of 
        runs in his career, and also holds the record for the most international sixes across all formats.""", 
        metadata={"source":"test"}),
    
        Document(
        page_content="""The current captain, a right-handed batsman with a high score of 
        runs in his career, and also holds the record for the most international sixes across all formats.""", 
        metadata={"source":"test"}),
 
        Document(
        page_content="""A legendary batsman who is the only player to score 100 international centuries and the highest run-scorer in test match cricket. """, 
        metadata={"source":"test"}),      
         
          
        Document(
        page_content="""A successful captain nicknamed "Captain Cool" for his calm demeanor and innovative strategies, he led India to victory in the 2011 World Cup and the 2007 T20 World Cup. """, 
        metadata={"source":"test"}), 
          
]



#add documents (har document ko id generate hota hai)
vector_store.add_documents(documents=docs)

#view documents
emb=vector_store.get(include=['embeddings','documents','metadatas'])

print(emb)


#similarity search
result = vector_store.similarity_search(query="Who has aggresive style ",k=1)


#similarity search with score
result_score = vector_store.similarity_search_with_score(
    query="Who is captain?",
    k=1
)


#meta-data filterng
result_meta_filtering=vector_store.similarity_search_with_score(
    query="",
    filter={"team":"india"}
)

#updating documents



for docs in result:
    print(docs.page_content)