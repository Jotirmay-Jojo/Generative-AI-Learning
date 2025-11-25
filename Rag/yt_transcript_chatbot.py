from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

ytt_api = YouTubeTranscriptApi()

video_id= "Gfr50f6ZBvo" 
try:
    transcript_list=YouTubeTranscriptApi().fetch(video_id, languages=['en'])

    # for entry in transcript_list:
    #     print(entry)

    transcript=" ".join([snippet.text for snippet in transcript_list])

    # print(transcript)

except TranscriptsDisabled:
    print("No transcript availabe for this video")
    

#Text Splitting

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.create_documents([transcript])

print(len(chunks))


embeddings = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
vector_store= FAISS.from_documents(chunks,embeddings)

# print(vector_store.index_to_docstore_id)

# STAGE 2 

#retirve the query
retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={"k": 4})

# result=retriever.invoke('what is deepmind')

# print(result)

#STAGE 3 : AUGMENTATION 

model = ChatGroq(model='llama-3.1-8b-instant')

prompt =PromptTemplate(
    template="""You are a helpful assistant.
        Answer only from the provided transcript context.
        if the context is insufficient, just say you dont know.
        {context}
        Question: {question}""",
    input_variables=['context','question']
)

question ="is the topic of aliens discussed in the video? if yes what was discussed"

retrieved_docs = retriever.invoke(question)
context= "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt=prompt.invoke({'context': context,'question':question})

print(final_prompt)

answer = model.invoke(final_prompt)
print(answer.content)


#Chain

