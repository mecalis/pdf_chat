from dotenv import load_dotenv, find_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain_classic.callbacks.manager import get_openai_callback
import os, certifi

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = certifi.where()

from tiktoken.load import load_tiktoken_bpe

local_path = "C:/ML/pdfllm/encodings/cl100k_base.tiktoken"
mergeable_ranks = load_tiktoken_bpe(local_path)

def main():
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings(openai_api_key="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")
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Kérdezz valamit a pdf-ről:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = ChatOpenAI(openai_api_key="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")
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
    

if __name__ == '__main__':
    main()