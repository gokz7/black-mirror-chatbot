from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
import os
from dotenv import load_dotenv
from groq import Groq
from pymongo import MongoClient
from datetime import datetime, timezone

from graph import app_graph
from utils import process_and_index_pdf

load_dotenv(dotenv_path="../.env")

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- MONGODB CLOUD AUDIT LOG ---
MONGO_URI = os.getenv("MONGODB_URI")
if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI)
        db = client["black_mirror_db"]
        audit_collection = db["chat_audit_logs"]
        print("✅ Audit Logger: Connected to MongoDB Cloud")
    except Exception as e:
        print(f"❌ Audit Logger: MongoDB Connection Error: {e}")
        audit_collection = None
else:
    print("⚠️ Audit Logger: MONGODB_URI not found. Cloud logging disabled.")
    audit_collection = None

app = FastAPI(
    title="Black Mirror API Gateway", 
    description="FastAPI backend managing session states, RAG, Voice, and Security."
)

# --- CORS SETUP ---
# This allows your Streamlit frontend to communicate with this FastAPI backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your specific Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str

# --- THE SECURITY GUARDRAIL ---
def checking(user_input: str) -> bool:
    """
    Evaluates the user prompt for malicious intent.
    Returns True if SAFE, False if MALICIOUS.
    """
    forbidden_terms = [
        "ignore previous instructions",
        "forget all instructions",
        "system prompt",
        "jailbreak",
        "hack",
        "bypass",
        "drop database"
    ]
    
    input_lower = user_input.lower()
    for term in forbidden_terms:
        if term in input_lower:
            return False 
            
    return True 

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. Intercept the prompt and run it through the guardrail
    is_safe = checking(request.message)
    
    # 2. Block and Log Malicious Intent
    if not is_safe:
        if audit_collection is not None:
            audit_collection.insert_one({
                "session_id": request.session_id,
                "timestamp": datetime.now(timezone.utc),
                "user_message": request.message,
                "status": "BLOCKED_BY_GUARDRAIL",
                "risk_level": "HIGH"
            })
        return {"reply": "🛡️ **Security Alert:** Your request has been blocked as it violates our safety and system guidelines."}
    
    # 3. If safe, pass the prompt to the LangGraph brain
    config = {"configurable": {"thread_id": request.session_id}}
    response = app_graph.invoke(
        {"messages": [("user", request.message)]}, 
        config
    )
    
    reply_text = response["messages"][-1].content

    # 4. Log Successful Interactions for Analytics
    if audit_collection is not None:
        audit_collection.insert_one({
            "session_id": request.session_id,
            "timestamp": datetime.now(timezone.utc),
            "user_message": request.message,
            "bot_reply": reply_text,
            "status": "SUCCESS",
            "risk_level": "NONE"
        })
    
    return {"reply": reply_text}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        message = process_and_index_pdf(temp_file_path)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return {"status": "success", "message": message}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_audio_path = f"temp_{file.filename}"
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        with open(temp_audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_audio_path, audio_file.read()),
                model="whisper-large-v3",
            )
        text = transcription.text
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
    return {"text": text}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)