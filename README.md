# 🪞 Black Mirror AI Chatbot

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas%20Vector-47A248)
![Groq](https://img.shields.io/badge/Groq-Llama%203%20%7C%20Whisper-F55036)

A conversational AI chatbot with document-aware (RAG) responses, voice input, persistent chat history, and safety guardrails — built with FastAPI, Streamlit, LangGraph, and Groq.

## ✨ Features

* **Conversational AI:** Powered by Groq's `llama-3.1-8b-instant`, orchestrated through LangGraph for stateful, multi-turn conversations.
* **Context-Aware Memory:** Each conversation session is tracked via a LangGraph checkpointer, so the model retains context across turns within a session.
* **Retrieval-Augmented Generation (RAG):** Users can upload a PDF, which is chunked, embedded, and stored in MongoDB Atlas Vector Search. Relevant chunks are retrieved and injected into the model's context when answering questions.
* **Security Guardrail Middleware:** Incoming messages are screened for prompt-injection and system-prompt-extraction attempts (e.g., "ignore previous instructions", "jailbreak") before reaching the LLM. Flagged requests are blocked and logged instead of processed.
* **Voice Input:** Audio is recorded in-browser and transcribed via Groq's Whisper (`whisper-large-v3`) API, then routed through the same chat pipeline as typed input.
* **Conversation History:** Past sessions are listed in a sidebar, can be reopened or deleted, and are persisted locally between runs.
* **Audit Logging:** Every chat interaction (and every blocked/malicious attempt) is logged to MongoDB Atlas with a timestamp and risk level for traceability.
* **Responsive UI:** Built with Streamlit's chat components, usable across desktop and mobile browsers.

## 🏗️ Architecture Flow

```text
User
 │
 ▼
Streamlit Frontend (frontend/app.py)
 │  - Chat UI, session/history management, audio recorder
 │  - Calls backend via HTTP (BACKEND_URL)
 ▼
FastAPI Backend (backend/main.py)
 │  - /chat      → guardrail check → LangGraph invoke → response
 │  - /upload    → PDF chunking + embedding → MongoDB Atlas Vector Search
 │  - /transcribe→ Groq Whisper transcription
 ▼
LangGraph Orchestrator (backend/graph.py)
 │  - Stateful graph with MemorySaver checkpointer (per-session thread_id)
 │  - Retrieves relevant document chunks (RAG) when available
 │  - Trims conversation history to stay within the model's token budget
 │  - Sends system prompt + trimmed history + retrieved context to Groq
 ▼
Groq LLM (llama-3.1-8b-instant)
```

**Supporting services:**
- **MongoDB Atlas** — stores audit logs (`chat_audit_logs` collection) and, separately, PDF vector embeddings (`pdf_vectors` collection with a Vector Search index).
- **Groq API** — powers both the chat model and Whisper-based audio transcription.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, `audio_recorder_streamlit` |
| Backend | FastAPI, Uvicorn |
| Orchestration | LangGraph, LangChain |
| LLM | Groq (`llama-3.1-8b-instant`), Groq Whisper (`whisper-large-v3`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | MongoDB Atlas Vector Search |
| Database | MongoDB Atlas (audit logs) |
| Deployment | Render (backend + frontend as separate services) |

## ⚙️ How It Works — Request Flow

1. User sends a message (typed or voice) from the Streamlit UI.
2. The frontend sends the message to the FastAPI `/chat` endpoint.
3. The **guardrail** (`checking()`) scans the input for known prompt-injection patterns. If flagged, the request is blocked, logged as high-risk, and a safe default response is returned — no LLM call is made.
4. If safe, the message is passed into the **LangGraph** state graph, keyed by a per-session `thread_id` so conversation history is maintained correctly across turns.
5. If a PDF has been uploaded and indexed, the graph retrieves the top matching chunks from MongoDB Atlas Vector Search (`pdf_vectors` collection) and injects them into the system prompt as context.
6. The full message history is trimmed to fit the model's token budget before being sent to Groq.
7. Groq generates a response, which is returned to the frontend, displayed, and saved to local session history.
8. The interaction (or the block event, if guardrail-triggered) is logged to MongoDB (`chat_audit_logs` collection) for auditing.

## 🚀 Getting Started (Local Development)

### 1. Prerequisites
You will need Python 3.10+, a Groq API Key, and a MongoDB Atlas Cluster.

### 2. Environment Variables
Create a `.env` file in the `backend/` directory:

```
GROQ_API_KEY="your_groq_api_key"
MONGODB_URI="mongodb+srv://<user>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority"
```

Create a `.env` file in the `frontend/` directory (only needed if pointing at a deployed backend rather than localhost):

```
BACKEND_URL="http://localhost:8000"
```

### 3. MongoDB Vector Search Setup
In your MongoDB Atlas dashboard, create a Search Index named `vector_index` on the **`pdf_vectors`** collection using this JSON configuration:

```json
{
  "fields": [
    {
      "numDimensions": 384,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

### 4. Run the Servers
Open two terminal windows:

**Terminal 1 (Backend):**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Terminal 2 (Frontend):**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```
AI-CHATBOT-MVP/
├── backend/
│   ├── main.py          # FastAPI app: /chat, /upload, /transcribe endpoints, guardrail logic
│   ├── graph.py          # LangGraph state graph: memory, RAG retrieval, LLM invocation
│   ├── utils.py           # PDF processing, chunking, embedding, MongoDB vector store
│   └── requirements.txt
├── frontend/
│   ├── app.py             # Streamlit UI: chat, sidebar history, voice input, PDF upload
│   └── requirements.txt
└── README.md
```
