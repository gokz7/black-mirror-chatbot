import streamlit as st
import requests
import uuid
import json
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
from audio_recorder_streamlit import audio_recorder

# --- 1. LOCAL HISTORY MANAGER ---
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_history(history_dict):
    # Keep only the last 3 chats to prevent the file from getting too large
    while len(history_dict) > 3:
        oldest_key = list(history_dict.keys())[0]
        del history_dict[oldest_key]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_dict, f)

st.set_page_config(page_title="Production AI Chatbot", page_icon="🤖", layout="centered")

st.title("Black mirror Chatbot")
st.caption("Advanced multi-modal AI architecture powered by Llama 3.1.")

# --- 2. INITIALIZE STATE & LOAD HISTORY ---
if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_history()

# If no session exists, load the most recent one, or create a new one
if "session_id" not in st.session_state:
    if st.session_state.all_chats:
        latest_session = list(st.session_state.all_chats.keys())[-1]
        st.session_state.session_id = latest_session
        st.session_state.messages = st.session_state.all_chats[latest_session]
    else:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.all_chats[st.session_state.session_id] = []

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

# --- 3. SIDEBAR: CHAT HISTORY, RAG & VOICE ---
with st.sidebar:
    st.header("💬 Chat History")

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.session_id = new_id
        st.session_state.messages = []
        st.session_state.all_chats[new_id] = []
        save_history(st.session_state.all_chats)
        st.rerun()

    # Display past chats with Delete functionality
    st.write("**Recent Conversations:**")

    # Convert to a list so we can safely delete items while iterating
    for s_id in list(reversed(list(st.session_state.all_chats.keys()))):
        msgs = st.session_state.all_chats[s_id]

        # Create a preview title
        if msgs:
            preview = msgs[0]["content"][:20] + "..." if len(msgs[0]["content"]) > 20 else msgs[0]["content"]
        else:
            preview = "Empty Chat"

        # Create two columns: 80% for the chat button, 20% for the delete button
        col1, col2 = st.columns([0.8, 0.2])

        with col1:
            if st.button(preview, key=f"load_{s_id}", use_container_width=True):
                st.session_state.session_id = s_id
                st.session_state.messages = msgs
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"del_{s_id}"):
                # 1. Delete from memory and save
                del st.session_state.all_chats[s_id]
                save_history(st.session_state.all_chats)

                # 2. If we just deleted the active chat, reset the screen
                if st.session_state.session_id == s_id:
                    if st.session_state.all_chats:
                        latest_session = list(st.session_state.all_chats.keys())[-1]
                        st.session_state.session_id = latest_session
                        st.session_state.messages = st.session_state.all_chats[latest_session]
                    else:
                        new_id = str(uuid.uuid4())
                        st.session_state.session_id = new_id
                        st.session_state.messages = []
                        st.session_state.all_chats[new_id] = []
                st.rerun()

    st.divider()

    st.header("📄 Document Upload (RAG)")
    uploaded_file = st.file_uploader("Upload a PDF to give the AI context", type="pdf")
    if uploaded_file is not None:
        if st.button("Process & Index PDF"):
            with st.spinner("Chunking and generating embeddings..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{BACKEND_URL}/upload", files=files)
                if response.status_code == 200:
                    st.success("PDF successfully indexed!")
                else:
                    st.error(f"Backend HTTP Error: {response.status_code}")

    st.divider()

    st.header("🎙️ Voice Input")
    st.write("Click the mic to speak, click again to stop.")
    audio_bytes = audio_recorder(
        text="",
        icon_size="2x",
        icon_name="microphone",
        recording_color="#00FF00",
        neutral_color="#6c757d"
    )

# --- 4. RENDER CURRENT CHAT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 5. INPUT ROUTING ---
text_prompt = st.chat_input("Type your message here...")
final_prompt = None

if text_prompt:
    final_prompt = text_prompt
elif audio_bytes and audio_bytes != st.session_state.last_audio:
    st.session_state.last_audio = audio_bytes
    with st.spinner("Transcribing audio..."):
        files = {"file": ("voice.wav", audio_bytes, "audio/wav")}
        voice_res = requests.post(f"{BACKEND_URL}/transcribe", files=files)
        if voice_res.status_code == 200:
            final_prompt = voice_res.json()["text"]
            st.sidebar.success(f"Heard: {final_prompt}")
        else:
            st.sidebar.error("Failed to transcribe audio.")

# --- 6. SEND TO BACKEND & SAVE ---
if final_prompt:
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    # Save after user input
    st.session_state.all_chats[st.session_state.session_id] = st.session_state.messages
    save_history(st.session_state.all_chats)

    with st.chat_message("user"):
        st.markdown(final_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"message": final_prompt, "session_id": st.session_state.session_id},
                    timeout=30
                )
                if response.status_code == 200:
                    reply = response.json()["reply"]
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    # Save after AI reply
                    st.session_state.all_chats[st.session_state.session_id] = st.session_state.messages
                    save_history(st.session_state.all_chats)
                else:
                    st.error(f"Backend HTTP Error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to reach the backend gateway.")