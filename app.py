import streamlit as st
from rag import get_answer  # Import the function from rag.py

# Title of your app
st.title("Tech Mentor Chatbot")

# User input field
user_input = st.text_input("Ask me anything about tech mentorship:")

if user_input:
    answer = get_answer(user_input)  # Get response from RAG
    st.write("**Bot:**", answer)
