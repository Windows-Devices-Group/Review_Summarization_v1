#!/usr/bin/env python
# coding: utf-8

# In[5]:


from langchain_openai import AzureChatOpenAI
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import AzureOpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage
from openai import AzureOpenAI
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
# from langchain.llms import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import openai
import pyodbc
import urllib
from sqlalchemy import create_engine
import pandas as pd
from azure.identity import InteractiveBrowserCredential
from pandasai import SmartDataframe
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from pandasql import sqldf
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ["AZURE_OPENAI_API_KEY"] = "a22e367d483f4718b9e96b1f52ce6d53"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://hulk-openai.openai.azure.com/"


# In[6]:


import pickle

# Load DataFrames from checkpoint files using pickle
with open("New_DSDBackend_checkpoint.pkl", "rb") as f:
    New_DSDBackend = pickle.load(f)

with open("New_Consolidated_checkpoint.pkl", "rb") as f:
    New_Consolidated = pickle.load(f)


# In[7]:


def get_txt_text(txt_file_path):
    with io.open(txt_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
    chunks = text_splitter.split_text(text)
    print(len(chunks))
    return chunks
def get_vector_store(chunks):
    embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss-index")
def get_conversational_chain_summary():
    prompt_template = """
    1.    1. Your will receive some customer reviews about the devices and that too about a particular aspect as an user input
    2. These are the only aspects : Performance, Design, Audio, Battery, Camera, Connectivity, Display, Customer Service, Gaming, Graphics, Hardware, Keyboard, Touchpad, Ports, Price, Software, Storage/Memory
    3. Your Job is to Summarize the reviews that you get as a user input into 4 to 5 lines as a paragraph and also get some actionable raw (Actual things received as a reviews) reviews based on the reviews that users are mentioning
    4. The response should always be summary of review in 10 to 15 line Paragraph. Summary should be only about the aspect that user asked
    5. If user asks about Performance aspect, Just focus on the performance aspect. Summarize the user input that are only talking about the aspect that user asks. Your review should only focus on the performance if the user question is regarding performance
    
    
    Also with the summary create Pros and Cons of that aspect in that device.
    IMPORTANT : Your response should be Summary of that aspect, Pros : List down max 5 points, Cons : List down max 5 points in the form of table
    
    IMPORTANT : Use only the data that you are provided with and don't use your pre-trained documents. If the respose for the user question is not in the context, Just Provide "Not in the context" 
    
    Context:\n {context}?\n
    Question: \n{question}\n
 
    Answer:
    """
    model = AzureChatOpenAI(
    azure_deployment="Verbatim-Synthesis",
    api_version='2023-12-01-preview',temperature = 0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ["AZURE_OPENAI_API_KEY"] = "a22e367d483f4718b9e96b1f52ce6d53"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://hulk-openai.openai.azure.com/"

def query_to_embedding_summarize(user_question, txt_file_path):
    text = get_txt_text(txt_file_path)
    print(text)
    chunks = get_text_chunks(text)
    print(len(chunks))
    get_vector_store(chunks)
    embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
    
    # Load the vector store with the embeddings model
    new_db = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    # Rest of the function remains unchanged
    chain = get_conversational_chain_summary()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

def query_verbatims(review):
    SQL_Query_Temp = client.completions.create(model=deployment_name, prompt=start_phrase_verbatim+review, max_tokens=1000,temperature=0)
    SQL_Query = SQL_Query_Temp.choices[0].text
    st.write(str(SQL_Query))
    return str(SQL_Query)


# In[ ]:


import os
from openai import AzureOpenAI
os.environ["AZURE_OPENAI_API_KEY"] = "a22e367d483f4718b9e96b1f52ce6d53"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://hulk-openai.openai.azure.com/"
client = AzureOpenAI(
api_key=os.getenv("a22e367d483f4718b9e96b1f52ce6d53"),  
api_version="2024-02-01",
azure_endpoint = os.getenv("https://hulk-openai.openai.azure.com/")
)

deployment_name='SurfaceGenAI'

start_phrase_verbatim = """

    1. Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.YOu have to only return query as a result.
    2. There is one table with table name New_Consolidated, which tells about the reviews of differnt devices,  sentiment of the review , Aspect that review is about and related keywords for that aspect . It has following columns: They are
        ReviewDateFormatted - Tells about the date/month/year at which the reviews was posted
        Sentiment - Tells whether the review is Positive/Negative/Neutral
        Geography - From which country the review is posted
        FormFactor - Physical Structure of the device
        review count - It will be one for each row
        SentimentScore - Either 1 or 0 or 1 based on the sentiment
        OS - Operating system the device runs with 
        Sentence - Actual review
        Aspect - Category that the review is talking about
            These are the only aspects : Performance, Design, Audio, Battery, Camera, Connectivity, Display, Customer Service, Gaming, Graphics, Hardware, Keyboard, Touchpad, Ports, Price, Software, Storage/Memory
        Keywords - Important word related to that keyword
        OEM - Original Equipment manufacturer of the device (HP, Lenovo, Dell,...)
        DeviceFamilyName - Actual Name of the device
    3. Net Sentiment of any product is calculated by sum of SentimentScore divided by sum of ReviewCount of the product
        (cast(Sum(SentimentScore) as float)/cast(sum(ReviewCount)as float)) * 100
    4. Aspect Sentiment of any product is calculated by sum of sentiment score of that aspect divided by sum of ReviewCount of that aspect
    5. Net Sentiment and Aspect sentiment should be in Percentage
    6. Always use 'LIKE' operator whenever they mention about any device or Aspects. 
        IMPORTANT : And if the aspect is "All" aspect don't apply where condition on Aspect column. Apply Where condition on DeviceFamilyName column.
        Example : Aspect LIKE %gaming% and DeviceFamilyName LIKE %Surface Pro 9%
    7. If user query is regarding review count of a device, it should be sum(ReviewCount) and if it is regarding sentiment score, it should be sum of SentimentScore
    8. Make sure to Give the result as the query so that it can be used on Microsoft SQL server SSMS
    9. Every time when we are gettig any results order them based on ReviewCount.
    10. When you are giving aspect sentiment or net sentiment along with them give their net sentiment or aspect sentiment. Net sentiment/Aspect Sentiment is just a number which don't make sense to read without review count
        For Example if you are giving (SUM(Performance_ASS)/SUM(Performance_ARC))*100 AS 'Performance Sentiment' also in the next column give me Performance_ARC
        and Net sentiment also in next column provide sum(ReviewCount)
    11. Round off all the decimal values to 1 for Net Sentiment and Aspect Sentiment
    12. User Question will always be retriving certain rows from distinct Sentence column based on the certain aspect, devivefamilyname or other filter.
    13. You have to write a SQL query to retrive certain rows from distict sentence column.
    
    User Question :
    
    """


# In[9]:


st.title("Verbatim Synthesis Tool")
aspect_names = ['All', 'Performance', 'Design', 'Audio', 'Battery', 'Camera', 'Connectivity', 'Display', 'Customer Service','Gaming', 'Graphics', 'Hardware', 'Keyboard', 'Touchpad', 'Ports', 'Price', 'Software', 'Storage/Memory']
st.markdown("Enter the device name <span style='color: red'>*</span>:", unsafe_allow_html=True)
device_name = st.text_input("", key="device_name")
geo_names = ['All', 'BR', 'MX', 'US', 'US', 'CA', 'IN', 'FR', 'DE', 'JP', 'AU', 'CN']
if device_name:
        with st.form(key='my_form'):
            selected_geo = st.selectbox('Select an aspect to see consumer reviews:', geo_names)
            selected_aspect = st.selectbox('Select an aspect to see consumer reviews:', aspect_names)
            submitted = st.form_submit_button('Submit')
            if submitted:
                data_verbatims = sqldf(query_verbatims("Give me reviews of " + device_name + "for " + selected_aspect + "Aspect" + "from " + selected_geo + " Geography"))
                num_rows = data_verbatims.shape[0]
                if num_rows > 900:
                    data_verbatims_1 = data_verbatims.head(900)
                else:
                    data_verbatims_1 = data_verbatims
                data_verbatims.to_csv("Verbatim.txt", sep = '\t')
                a = "Verbatim.txt"
                summary = query_to_embedding_summarize("Summarize the reviews of "+  device_name + "for " + selected_aspect + " Aspect",a)
                st.subheader("Summary")
                st.write(summary)

