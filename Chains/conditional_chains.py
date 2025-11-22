#positive negative response analysis and generate conditiona; response using chains
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class Person(BaseModel):
    sentiment:Literal['positive','negative']=Field(description="If the feedback is positive chosse positve literal else choose negative literal ")


parser = PydanticOutputParser(pydantic_object=Person)
parser2 = StrOutputParser()

prompt = PromptTemplate(
    template="Take the user feedback and determine if it's positive or negative.\nFeedback: {feedback}\n{format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="Write an 2 lines appropriale responce to this positve feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an 2 lines appropriale responce to this negative feedback \n {feedback}",
    input_variables=['feedback']
)


classifier_chain = prompt | model |parser
branch_chain = RunnableBranch(
    
    (lambda x:x.sentiment=='positive',prompt2|model|parser2),
    (lambda x:x.sentiment=='negative',prompt3|model|parser2),
    RunnableLambda(lambda x:"Couldnt get the sentiment")
)

chain = classifier_chain | branch_chain
result=chain.invoke({'feedback':"This is wonderful device"})
print (result)