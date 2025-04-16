import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## CHAT WITH KNOWLEDGE BASE

This chatbot uses the Retrieval-Augmented Generation (RAG) framework powered by open-source Hugging Face models. It processes uploaded PDF documents, chunks the content, stores it in a vector store, and provides accurate answers using LLMs hosted on Hugging Face.

### How It Works

1. **Enter Your Hugging Face API Key**: Get one from https://huggingface.co/settings/tokens  
2. **Upload Your Documents**: Supports multiple PDFs  
3. **Ask a Question**: You'll get context-aware answers from your uploaded files
""")

# Hugging Face API Key
hf_api_key = st.text_input("Enter your Hugging Face API Key:", type="password", key="hf_api_key")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(hf_api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context", and do not make anything up.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",
        model_kwargs={"temperature": 0.3, "max_length": 512},
        huggingfacehub_api_token=hf_api_key,
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, hf_api_key):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(hf_api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

def main():
    st.header("AI Chatbot")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and hf_api_key:
        user_input(user_question, hf_api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on Submit & Process", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and hf_api_key:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
