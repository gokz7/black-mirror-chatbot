from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import os

def process_and_index_pdf(pdf_file_path):
    """
    Takes a PDF file, extracts the text, chunks it into manageable pieces,
    converts them to dense vectors, and stores them in MongoDB Atlas.
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
    
    # 4. Build the MongoDB Vector Database
    client = MongoClient(os.getenv("MONGODB_URI"))
    collection = client["black_mirror_db"]["chat_audit_logs"]
    
    MongoDBAtlasVectorSearch.from_documents(
        documents=texts,
        embedding=embeddings,
        collection=collection,
        index_name="vector_index"
    )
    
    return "PDF successfully indexed into MongoDB Atlas vector database."

def get_faiss_retriever():
    """
    Loads the MongoDB Vector Search so the LangGraph AI can search it.
    (Kept the function name as 'get_faiss_retriever' so it doesn't break graph.py!)
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    
    client = MongoClient(os.getenv("MONGODB_URI"))
    collection = client["black_mirror_db"]["chat_audit_logs"]
    
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index"
    )
    
    # Return a retriever configured to fetch the top 3 most relevant chunks
    return vector_store.as_retriever(search_kwargs={'k': 3})