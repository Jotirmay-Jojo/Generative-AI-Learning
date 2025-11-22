
#Simple incident responce 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.3)


class Incident(BaseModel):
    attack_type: str =Field(description="Likely an cyber attack")
    risk_score:int = Field(ge=1,le=10,description="How dangerous this seem 1 low 10 critical")
    evidence: list[str] = Field(description="Short bullet points with clues from the desciption/logs")
    reccomended_actions: list[str] =Field(description="concrete step a juniour analyst shoult take")


parser = PydanticOutputParser(pydantic_object=Incident)

prompt = PromptTemplate(
    template="""You are a cybersecurity incident response assistant.
                your job: 
                -Read the incident description
                -Decide the most likely attack_type
                -Give a risk_score between 1 to 10
                -Extract key Evidence as a short bullet points
                -Suggest clear reccomended actions.  
                Incident Descrption :
                {incident_description}\n {format_instructions}""",
    input_variables=['incident_description'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain = prompt | model | parser

result= chain.invoke({"incident_description":"Bruth force on SSH"})
print(result)