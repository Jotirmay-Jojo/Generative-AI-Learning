from langchain_core.prompts import PromptTemplate

#template
template = PromptTemplate(
    template="""
    Please Summarise the research paper tittled {paper_input} with the following specifications:
    Explaination style {style_input}
    Explainatino Length {length_input}
    1. Mathematical details : 
        - Include relevent mathematical equations if present in the paper
        - Exaplain the mathermatical concept useing simple, intutive code snipperts where applicable.
    2. Analogies :
        - Use relatable analogies to simplify comple ideas.
        if certain information is not available in the paper, respond with : "Insufficient information available" instead f guessing
        Ensure the summary is clear, accurate and aligned with the provided style and length.
    """, 
    input_variables=['paper_input','style_input','length_input'],
    validate_template=True
    
)

template.save('template.json')