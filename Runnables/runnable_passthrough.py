from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnablePassthrough, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser=StrOutputParser()

prompt1= PromptTemplate(
    template="tell me a one line joke on {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Explain the provided joke {text}, in 2 lines",
    input_variables=['text']
)

joke_generator = RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    'joke': RunnablePassthrough(),
    'explain': RunnableSequence(prompt2,model,parser)
})

chain= RunnableSequence(joke_generator,parallel_chain)

result=chain.invoke({'topic':'my ex'})
print(result)


