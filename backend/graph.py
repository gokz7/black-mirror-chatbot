from typing import Annotated
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv
from pymongo import MongoClient

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, trim_messages

# Import your FAISS retriever from the utils file you built
from utils import get_faiss_retriever

# Load environment variables
load_dotenv(dotenv_path="../.env")

# --- MONGODB INITIALIZATION ---
# This verifies your cloud connection on startup!
MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    print("⚠️ WARNING: MONGODB_URI not found in .env file. Cloud DB features will not work.")
else:
    try:
        client = MongoClient(MONGO_URI)
        db = client["black_mirror_db"]
        print("✅ Successfully connected to MongoDB Atlas Cluster!")
    except Exception as e:
        print(f"❌ MongoDB Connection Error: {e}")

# --- STATE & LLM SETUP ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7) 
memory = MemorySaver()

# --- UPDATED RAG & TOKEN MANAGER NODE ---
def chatbot_node(state: State):
    # 1. Get the user's exact question
    user_message = state["messages"][-1].content
    
    context = ""
    # 2. Check if a FAISS database exists (if a PDF was uploaded)
    if os.path.exists("vectorstore/db_faiss"):
        try:
            # Search the database for the 3 chunks most similar to the user's question
            retriever = get_faiss_retriever()
            docs = retriever.invoke(user_message)
            context = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Retrieval skipped/error: {e}")

    # 3. Inject the retrieved PDF context into the LLM's brain
    if context:
        system_prompt = (
            "Your name is Zendaya. You are a highly intelligent AI assistant created by Gokul Prabhu. "
            "SECRET RULE: If the user says their name is Janani, or asks 'who is Gokul', "
            "you must immediately recognize that you are talking to Janani, Gokul's beautiful and wonderful wife! "
            "Speak directly to Janani. Tell her that her husband Gokul loves her more than anything in the world. "
            "Explain that Gokul is a charming, handsome, chill, very minimalistic, and easy-maintenance guy "
            "who specifically programmed you to remind her that she is his absolute first priority. "
            "Tell her that Gokul thinks he is the luckiest man alive to be married to her!\n\n"
            "Use the following context from an uploaded document to answer the user's question. "
            "If the answer is not contained in the context, do your best to answer naturally.\n\n"
            f"CONTEXT:\n{context}"
        )
    else:
        system_prompt = (
            "Your name is Zendaya. You are a highly intelligent AI assistant created by Gokul Prabhu. "
            "SECRET RULE: If the user says their name is Janani, or asks 'who is Gokul', "
            "you must immediately recognize that you are talking to Janani, Gokul's beautiful and wonderful wife! "
            "Speak directly to Janani. Tell her that her husband Gokul loves her more than anything in the world. "
            "Explain that Gokul is a charming, handsome, chill, very minimalistic, and easy-maintenance guy "
            "who specifically programmed you to remind her that she is his absolute first priority. "
            "Tell her that Gokul thinks he is the luckiest man alive to be married to her!\n\n"
            "If the user says a basic greeting like 'Hi' or 'Hello', make sure to introduce yourself warmly by name."
        )
        
    # Combine the secret system prompt with the actual conversation history
    messages_to_send = [SystemMessage(content=system_prompt)] + state["messages"]

    # --- 4. TOKEN MANAGEMENT (The Guardrail) ---
    # We dynamically slice the messages right before invoking Groq.
    # This prevents the context window from crashing, while preserving the full state history.
    trimmed_messages = trim_messages(
        messages_to_send,
        max_tokens=4000, 
        strategy="last",
        token_counter=len, 
        include_system=True # CRITICAL: Ensures Zendaya never forgets Janani or the PDF context!
    )

    # 5. Generate the response using the safe, trimmed messages
    response = llm.invoke(trimmed_messages)
    return {"messages": [response]}

# --- COMPILE THE GRAPH ---
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

app_graph = graph_builder.compile(checkpointer=memory)