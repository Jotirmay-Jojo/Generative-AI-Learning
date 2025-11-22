from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnableSequence,RunnablePassthrough

load_dotenv()

def word_counter(text):
    return len(text.split())

runnable_counter = RunnableLambda(word_counter)

model= ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser=StrOutputParser()

prompt1= PromptTemplate(
    template="tell me a one line joke on {topic}",
    input_variables=['topic']
)

sequential_chain=prompt1|model|parser

parallel_chain= RunnableParallel({
    'joke':RunnablePassthrough(),
    'wordcount':runnable_counter
})

chain=sequential_chain | parallel_chain
result=chain.invoke({'topic':'cricket'})
final_result="""{} \n Word count - {}""".format(result['joke'],result['wordcount'])
print(final_result)