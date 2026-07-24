from pymongo import MongoClient
import os

# Cache both the Mongo connection and the embeddings model so they're
# created once per running server, not once per request. A fresh
# MongoClient on a mongodb+srv:// URI triggers a new DNS SRV lookup every
# time, which is unnecessary overhead and an occasional source of slow
# or hanging requests under constrained resources.
_mongo_client = None
_embeddings_instance = None

def _get_mongo_client():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(
            os.getenv("MONGODB_URI"),
            serverSelectionTimeoutMS=5000  # fail fast with a clear error instead of hanging near 30s
        )
    return _mongo_client

def _get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
    return _embeddings_instance

def has_indexed_documents():
    """
    Cheap check using plain pymongo only — no LangChain, no embeddings,
    no torch. Lets graph.py decide whether RAG retrieval is even worth
    attempting before paying the cost of loading the embedding model.
    """
    client = _get_mongo_client()
    collection = client["black_mirror_db"]["pdf_vectors"]
    return collection.count_documents({}) > 0

def process_and_index_pdf(pdf_file_path):
    """
    Takes a PDF file, extracts the text, chunks it into manageable pieces,
    converts them to dense vectors, and stores them in MongoDB Atlas.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_mongodb import MongoDBAtlasVectorSearch

    # 1. Extract Text from PDF
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()

    # 2. Chunk the Text (Prevents overflowing the LLM's context window)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)

    # 3. Generate Embeddings using the cached model
    embeddings = _get_embeddings()

    # 4. Build the MongoDB Vector Database (dedicated collection, separate from audit logs)
    client = _get_mongo_client()
    collection = client["black_mirror_db"]["pdf_vectors"]

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
    Only call this after confirming has_indexed_documents() is True.
    """
    from langchain_mongodb import MongoDBAtlasVectorSearch

    embeddings = _get_embeddings()

    client = _get_mongo_client()
    collection = client["black_mirror_db"]["pdf_vectors"]

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index"
    )

    # Return a retriever configured to fetch the top 3 most relevant chunks
    return vector_store.as_retriever(search_kwargs={'k': 3})