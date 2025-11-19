from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt

import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model=ChatGroq(model_name="llama-3.1-8b-instant")

st.title("Enter your prompt")

paper_input = st.selectbox ("Select Reseach Paper Name",["Select...","Attention is all you Need","Bert: Pre-training of Deep Bidirectional Transformers","GPT-3: Language Models are Few-Shot Learners", "Defusion Models Beat GANs on Inage syntesis"])
style_input = st.selectbox ("Select Explaination Style",["Beginner_Friendly","Technizal","Code-Oriented","Mathematical"])
length_input= st.selectbox ("Select Explaination LEngth",["Short (1-2) paragraph","Medium(3-5) paragraph","Long(Detailed Explaination)"])


template=load_prompt('template.json')

#fill the placceholders


if(st.button("Summarize")):
    
    chain = template | model
    result=chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input})
    

    st.write(result.content)