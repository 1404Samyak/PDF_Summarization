import streamlit as st 
#No message history is required here as this is only for summarization so no use of st.session_state
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
langchain_api_key=os.getenv("LANGCHAIN_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="LangChain: Summarize Text From a PDF", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From a PDF")
st.subheader('Summarize Text')

model =ChatGroq(model="Llama3-8b-8192")

chunks_prompt = """
Please summarize the following excerpt from the uploaded document. Focus on capturing the key points and main ideas clearly:
Content: `{text}`
Concise Summary:
"""

map_prompt_template=PromptTemplate(input_variables=['text'],
                                    template=chunks_prompt)

final_prompt = '''
Create a detailed and well-structured summary of the entire document. Start with a motivational title, followed by a brief introduction to set the context. Then provide the key points in a clear and concise format with sufficient detail:
Document Content: {text}
'''

final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)

uploaded_file=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=False) #A single file at a time for text summarization
if uploaded_file:
    documents=[]
    temppdf=f"./temp.pdf"
    with open(temppdf,"wb") as file:
        file.write(uploaded_file.getvalue())
        file_name=uploaded_file.name

    loader=PyPDFLoader(temppdf)
    docs=loader.load()
    documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    final_docs= text_splitter.split_documents(documents)

if st.button("Summarize the above PDFs") and uploaded_file:
    summarize_chain=load_summarize_chain(llm=model,chain_type='map_reduce',map_prompt=map_prompt_template,combine_prompt=final_prompt_template,verbose=True)
    output=summarize_chain.run(final_docs)
    st.write(output)
else:
    st.write("Kindly upload some pdfs")

    

                    
    
