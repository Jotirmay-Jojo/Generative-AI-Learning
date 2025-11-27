from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv

load_dotenv()


#Load Document
video_id="QZXE2-sFWxQ"
yt_api=YouTubeTranscriptApi()
fetched_transcript=yt_api.fetch(video_id)
final_transcript = " ".join([snippet.text for snippet in fetched_transcript])


#Chuncks Creation
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

chunk= splitter.split_text(final_transcript)


#Embedding

embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
vector_store = Chroma(
    collection_name='my_collection',
    embedding_function=embeddings,
    persist_directory='./chromadb' #optional for local persistance
) 

vector_store.add_texts(chunk)


#Retrival 
retriver= vector_store.as_retriever(search_kwargs={"k":2})


#Augmentation


prompt = PromptTemplate(
    template=""" You are a helpful assistant 
    you are given a transcript for context and using that only answer the question, if you cant find 
    asnwer from the context simply say you dont know
    Context: {context} \n\n Question: {question}""",

    input_variables=['context','question']
)

# Helper to convert docs -> single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


#Model 

parser=StrOutputParser()

model = ChatGroq(model='llama-3.1-8b-instant')

chain2= (
    {
        'context': retriver|RunnableLambda(format_docs) ,
        'question' : RunnablePassthrough()
    }
)

chain= chain2|prompt|model|parser
result=chain.invoke('canyou summarize the video')

print(result)