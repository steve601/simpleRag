import streamlit as st
import os
from langchain.llms import HuggingFaceHub 
from langchain_huggingface import HuggingFaceEndpoint # for accessing huggingface models
from langchain_huggingface import HuggingFaceEmbeddings # embeding the documents in the vectorstore
from langchain_huggingface import ChatHuggingFace # chat model
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS,Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

hf_tkn = os.getenv("HF_TOKEN")
st.title('Hey Dear! Ask me anything about Kenyan constitution.')

input_text = st.text_area('Write a message...')

pdfloader = PyPDFLoader('Constitution of Kenya 2010.pdf')
docs = pdfloader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
texts = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.from_documents(texts,embedding=embeddings)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=hf_tkn
)

chat_model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_template(""" 
        Answer the following question based only on the provided context
        Think step by step before providing a detailed answer
        <context>
        {context}
        </context>
        Question: {input}""")

retriever = db.as_retriever()

retrieval_chain = (
                {"context":retriever,"input":RunnablePassthrough()}
                | prompt
                | chat_model
                | StrOutputParser()
                )

def capitalize_first_letter(response):
    return response[0].upper() + response[1:] if response else response

if st.button('Response'):
    response = retrieval_chain.invoke(input_text).replace("Based on the provided context, ", "")
    response = capitalize_first_letter(response)
    st.success(response)


