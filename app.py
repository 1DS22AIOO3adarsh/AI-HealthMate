import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import time
import random
import os
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceBgeEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

# Load environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medicalbot"

# Set up Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest", 
    temperature=0.4, 
    max_output_tokens=500,
    google_api_key=GEMINI_API_KEY
)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load embeddings
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to Pinecone Vector Store
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

def chat_with_gemini(query, chat_history):
    """Retrieve relevant documents and generate a clean response with conversation history."""
    docs = docsearch.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    history = "\n".join([f"User: {q}\nBot: {a}" for q, a in chat_history])
    prompt = f"{history}\nContext: {context}\nUser: {query}\nBot (Provide a concise and clear response):"
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else response

# Streamlit UI setup
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.markdown("""
    <style>
        body {background-color: #0E1117; color: #EAECEF;}
        .css-1aumxhk {background-color: #161B22; padding: 20px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI Chatbot")
st.markdown("A chatbot that retrieves information from Pinecone and generates responses using Gemini AI.")

# Chat history session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Display chat history
for role, msg in st.session_state.messages:
    align = "flex-end" if role == "You" else "flex-start"
    bg_color = "#2b313e" if role == "You" else "#475063"
    st.markdown(f"""
        <div style="display: flex; justify-content: {align}; margin: 10px;">
            <div style="background-color: {bg_color}; padding: 10px; border-radius: 10px; color: white; max-width: 70%;">
                {msg}
            </div>
        </div>
    """, unsafe_allow_html=True)

# User input section
user_input = st.text_input("Type your message:", st.session_state.user_input, key="user_input_field")
if st.button("Send") and user_input:
    with st.spinner("Thinking..."):
        chat_history = [(st.session_state.messages[i][1], st.session_state.messages[i+1][1]) for i in range(0, len(st.session_state.messages)-1, 2)]
        response = chat_with_gemini(user_input, chat_history)
        time.sleep(random.uniform(0.5, 1.5))  # Simulate typing delay
        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Bot", response))
        st.session_state.user_input = ""  # Clear input field
        st.rerun()