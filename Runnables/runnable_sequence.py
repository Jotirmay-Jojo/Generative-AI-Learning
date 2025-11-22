from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model= ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = StrOutputParser()

prompt1= PromptTemplate(
    template="tell me a one line joke on {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Explain the provided joke {text}",
    input_variables=['text']
)

chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result=chain.invoke({'topic':'AI'})
print(result)