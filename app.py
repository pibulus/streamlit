import streamlit as st
import requests
import os
import openai
import langchain
from dotenv import load_dotenv
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    download_loader,
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    ServiceContext
)
from langchain import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
Airtable_TOKEN = os.getenv("AIRTABLE_API_TOKEN")

openai.api_key = OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

AirtableReader = download_loader('AirtableReader')
reader = AirtableReader(Airtable_TOKEN)
documents = reader.load_data(table_id="tblP6LQxyOo7JBiJC", base_id="appUkRhauFCWrTrBd")

max_input_size = 4096
num_output = 256
max_chunk_overlap = 20

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.1, model_name="text-davinci-003", max_tokens=num_output))

index = GPTVectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

st.title("Streamlit Query App")

query = st.text_input("Enter your query:")
if st.button("Submit"):
    response = query_engine.query(query)
    st.write(response)
