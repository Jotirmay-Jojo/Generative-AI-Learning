from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


#schema

schema = [
    ResponseSchema(name='fact1', description="Fact1 about the topic\n"),
    ResponseSchema(name='fact2', description="Fact1 about the topic\n"),
    ResponseSchema(name='fact3', description="Fact1 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give 3 facts about {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain = template|model|parser

result=chain.invoke({'topic':'black holes'})

print(result)