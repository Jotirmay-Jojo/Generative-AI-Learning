from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda,RunnableParallel,RunnablePassthrough,RunnableSequence

load_dotenv()

prompt1=PromptTemplate(
    template="Write a detailed Report on the {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Summarize the following text \n{text}",
    input_variables=['text']
)

model=ChatGroq(model='llama-3.1-8b-instant')

parser=StrOutputParser()

report_chain= RunnableSequence(prompt1,model,parser)

branch_chain=RunnableBranch(
    (lambda x:len(x.split())>500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()    
)

final_chain= report_chain | branch_chain

result=final_chain.invoke({'topic':'Russia VS Ukraine'})
print(result)
