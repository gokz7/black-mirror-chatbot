# Black Mirror Chatbot

A conversational AI chatbot with document-aware (RAG) responses, voice input, persistent chat history, and safety guardrails — built with FastAPI, Streamlit, LangGraph, and Groq.

## Features

- **Conversational AI** powered by Groq's `llama-3.1-8b-instant`, orchestrated through LangGraph for stateful, multi-turn conversations
- **Context-aware memory** — each conversation session is tracked via a LangGraph checkpointer, so the model retains context across turns within a session
- **Retrieval-Augmented Generation (RAG)** — users can upload a PDF, which is chunked, embedded, and stored in MongoDB Atlas Vector Search; relevant chunks are retrieved and injected into the model's context when answering questions
- **Security guardrail middleware** — incoming messages are screened for prompt-injection and system-prompt-extraction attempts (e.g. "ignore previous instructions", "jailbreak") before reaching the LLM; flagged requests are blocked and logged instead of processed
- **Voice input** — audio is recorded in-browser and transcribed via Groq's Whisper (`whisper-large-v3`) API, then routed through the same chat pipeline as typed input
- **Conversation history** — past sessions are listed in a sidebar, can be reopened or deleted, and are persisted locally between runs
- **Audit logging** — every chat interaction (and every blocked/malicious attempt) is logged to MongoDB Atlas with a timestamp and risk level, for traceability
- **Responsive UI** — built with Streamlit's chat components, usable across desktop and mobile browsers

## Architecture

```
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
- **MongoDB Atlas** — stores audit logs and, separately, PDF vector embeddings (Vector Search index)
- **Groq API** — powers both the chat model and Whisper-based audio transcription

## Tech Stack

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

## How It Works — Request Flow

1. User sends a message (typed or voice) from the Streamlit UI.
2. The frontend sends the message to the FastAPI `/chat` endpoint.
3. The **guardrail** (`checking()`) scans the input for known prompt-injection patterns. If flagged, the request is blocked, logged as high-risk, and a safe default response is returned — no LLM call is made.
4. If safe, the message is passed into the **LangGraph** state graph, keyed by a per-session `thread_id` so conversation history is maintained correctly across turns.
5. If a PDF has been uploaded and indexed, the graph retrieves the top matching chunks from MongoDB Atlas Vector Search and injects them into the system prompt as context.
6. The full message history is trimmed to fit the model's token budget before being sent to Groq.
7. Groq generates a response, which is returned to the frontend, displayed, and saved to local session history.
8. The interaction (or the block event, if guardrail-triggered) is logged to MongoDB for auditing.

## Environment Variables

| Variable | Used By | Purpose |
|---|---|---|
| `GROQ_API_KEY` | backend | Chat completions and audio transcription via Groq |
| `MONGODB_URI` | backend | Connection to MongoDB Atlas (audit logs + vector store) |
| `BACKEND_URL` | frontend | Public URL of the deployed FastAPI backend |

## Deployment

Deployed as two independent services on Render:
- **Backend** (`backend/`) — FastAPI app served via Uvicorn
- **Frontend** (`frontend/`) — Streamlit app, configured with `BACKEND_URL` pointing at the backend service

## Project Structure

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
