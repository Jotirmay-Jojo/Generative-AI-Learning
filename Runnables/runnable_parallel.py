from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser= StrOutputParser()

prompt1=PromptTemplate(
    template="Genarate a linked in post for my {topic} keep it short and crispy ",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Generate a twitter post for my {topic} keeping it short and noticable",
    input_variables=['topic']
)

chain = RunnableParallel({
    'linkedin': RunnableSequence(prompt1,model,parser),
    'tweet': RunnableSequence(prompt2,model,parser)
})

result = chain.invoke({'topic':'CEH module 5'})
final_result="""Linked in POST \n{}\n\n\n {} """.format(result['linkedin'],result['tweet'])
print(final_result)