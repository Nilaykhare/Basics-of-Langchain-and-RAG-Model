import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.llms import Ollama
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

## Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please answer the question asked."),
    ("user", "Question: {question}")
])

st.title("Langchain demo with Gemma:2b model")

input_text = st.text_input("What Question you have in mind")

## Ollama Gemma:2b model
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({
        "question": input_text,
    }))
