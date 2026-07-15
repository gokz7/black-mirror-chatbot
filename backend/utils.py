from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Define where the local FAISS database will be saved
DB_FAISS_PATH = 'vectorstore/db_faiss'

def process_and_index_pdf(pdf_file_path):
    """
    Takes a PDF file, extracts the text, chunks it into manageable pieces,
    converts them to dense vectors, and stores them in a local FAISS index.
    """
    # 1. Extract Text from PDF
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    
    # 2. Chunk the Text (Prevents overflowing the LLM's context window)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    
    # 3. Generate Embeddings using a fast, free local transformer model
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    
    # 4. Build the FAISS Vector Database
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    return "PDF successfully indexed into FAISS vector database."

def get_faiss_retriever():
    """
    Loads the saved FAISS database so the LangGraph AI can search it.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    # allow_dangerous_deserialization=True is required for local FAISS loading in newer LangChain versions
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Return a retriever configured to fetch the top 3 most relevant chunks
    return db.as_retriever(search_kwargs={'k': 3})