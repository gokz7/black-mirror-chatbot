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

# Import your MongoDB Atlas retriever from the utils file you built
from utils import get_faiss_retriever

# Load environment variables (from .env locally; no-op on Render, which injects env vars directly)
load_dotenv()

# --- MONGODB INITIALIZATION ---
# This verifies your cloud connection on startup!
MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    print("⚠️ WARNING: MONGODB_URI not found. Cloud DB features will not work.")
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

# --- RAG & TOKEN MANAGER NODE ---
def chatbot_node(state: State):
    # 1. Get the user's exact question
    user_message = state["messages"][-1].content

    # 2. Attempt retrieval from MongoDB Atlas Vector Search (empty context if nothing indexed yet)
    context = ""
    try:
        retriever = get_faiss_retriever()
        docs = retriever.invoke(user_message)
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Retrieval skipped/error: {e}")

    # 3. Build the system prompt, with retrieved context if any was found
    if context:
        system_prompt = (
            "You are Zendaya, a helpful, knowledgeable AI assistant. "
            "Answer clearly and conversationally.\n\n"
            "Use the following context from an uploaded document to answer the user's question. "
            "If the answer is not contained in the context, do your best to answer naturally.\n\n"
            f"CONTEXT:\n{context}"
        )
    else:
        system_prompt = (
            "You are Zendaya, a helpful, knowledgeable AI assistant. "
            "Answer clearly and conversationally.\n\n"
            "If the user says a basic greeting like 'Hi' or 'Hello', introduce yourself warmly by name."
        )

    # Combine the system prompt with the actual conversation history
    messages_to_send = [SystemMessage(content=system_prompt)] + state["messages"]

    # --- 4. TOKEN MANAGEMENT ---
    # Dynamically slice the messages right before invoking Groq.
    # This prevents the context window from overflowing, while preserving the full state history.
    trimmed_messages = trim_messages(
        messages_to_send,
        max_tokens=4000,
        strategy="last",
        token_counter=len,
        include_system=True
    )

    # 5. Generate the response using the trimmed messages
    response = llm.invoke(trimmed_messages)
    return {"messages": [response]}

# --- COMPILE THE GRAPH ---
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

app_graph = graph_builder.compile(checkpointer=memory)