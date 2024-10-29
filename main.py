import os
import json
import streamlit as st
from langchain_ollama import OllamaLLM

# Getting the model
model = OllamaLLM(model="llama3")

# Configuring openai api key
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
HF_API_KEY = config_data["HF_API_KEY"]

# Configuring streamlit page
st.set_page_config(
    page_title="LLaMA 3 ChatBot",
    page_icon="💬",
    layout="centered"
)

# Initialize chat session in streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit page title
st.title("💬 LLaMA 3 ChatBot")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user's message
user_prompt = st.chat_input("Ask LLaMA 3!")
if user_prompt:
    # Add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Send user's message to LLaMA 3 and get a response
    bot_response = model.invoke(input=user_prompt)  # Invoke the model with user input

    # Show the bot's response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
