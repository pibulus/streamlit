# Import required libraries
import os
import openai
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    download_loader,
    GPTKeywordTableIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
)
from langchain import OpenAI
import streamlit as st

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
Airtable_TOKEN = os.getenv("Airtable_TOKEN")

openai.api_key = OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

AirtableReader = download_loader('AirtableReader')
reader = AirtableReader(Airtable_TOKEN)
documents = reader.load_data(table_id="tblP6LQxyOo7JBiJC",base_id="appUkRhauFCWrTrBd")

# Define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.1, model_name="text-davinci-003", max_tokens=256))

# Build index
index = GPTVectorStoreIndex.from_documents(documents)

# Setup Streamlit
st.title('My Streamlit App')

# Text input
user_input = st.text_input("Enter your query here:")

# Button to execute query
if st.button('Run Query'):
    query_engine = index.as_query_engine()
    response = query_engine.query(user_input)
    st.write(response)
