from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm =HuggingFaceEndpoint(
    repo_id="WeiboAI/VibeThinker-1.5B",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

#1 Prompt -> detailed report 
template1 = PromptTemplate (
    template="Write a detailed report om {topic}",
    input_variables=["topic"]
)

#2 Prompt -> summary
template2 = PromptTemplate (
    template="Write a 5 line summary on the following {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({'topic':'black hole'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})
result=model.invoke(prompt2)

print(result)