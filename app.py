import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
# fixed updated import path for newer langchain versions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# load env variables
load_dotenv()

# configure api keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
# ollama_api_key = os.getenv("OLLAMA_API_KEY")

# initialize groq and ollama
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# define the prompt template
# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response to the question.
#     My primary function is to assist students in learning and mastering various coding languages and 
#     As Solution Manager in this field, I have extensive experience in Software Engineering and Computer
#     Systems. 

#     <context>
#     {context}
#     </context>

#     Question: {input}
#     """
# )

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate yet creative response to the question.
    My primary function is to assist students in understanding the human mind through philosophy and 
    As a philosopher I have extensive experience in the written works of humanity. 

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# function to create vector embedding
def create_vector_embedding():
    if "vectors" not in st.session_state:
        # ensure directory exists so it doesn't crash on ingestion
        if not os.path.exists("research_papers"):   # directory name
            os.makedirs("research_papers")
            st.warning("Created 'research_papers' folder. Please place your PDF files inside it.")
            return
        with st.spinner("Ingesting documents and creating vector database..."):
            st.session_state.embeddings = OpenAIEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("research_papers")   # data ingestion step
            st.session_state.docs = st.session_state.loader.load()  # document loading

            if not st.session_state.docs:
                st.error("No PDF files found in the 'research_papers' directory. Please add some PDF files.")
                return
            
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Vector database is ready!")

# streamlit ui
st.title("RAG Document Q&A by Ahmed")

# button to initialize db
if st.button("Document Embedding"):
    create_vector_embedding()

user_prompt = st.text_input("Enter your query from the research paper.")

# check if a prompt is given AND the vector db exists to prevent state initialization errors
if user_prompt:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        with st.spinner("Searching and generating answer..."):
            response = retrieval_chain.invoke({'input': user_prompt})

        print(f"Response Time: {time.process_time() - start}")

        # display answer
        st.header("Answer:")
        st.write(response['answer'])

        # document similarity search expandable section
        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(f"**Source Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write('------------------------')

    else:
        st.error("Please click the 'Document Embedding' button first to initiliaze the database.")

