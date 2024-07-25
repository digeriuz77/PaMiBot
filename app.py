import streamlit as st
import openai
from openai import OpenAI
import time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import random
import json
from datetime import datetime
import os
import base64

# Initialize logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Initialize NLTK
nltk.download('vader_lexicon', quiet=True)

# Streamlit configuration
st.set_page_config(page_title="✨ VHL Physical Activity Coachbot", layout="wide")

# Custom CSS (unchanged)
st.markdown("""
<style>
    .user-message {
        background-color: white;
        color: black;
        border-radius: 20px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 70%;
        align-self: flex-start;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .assistant-message {
        background-color: #007bff;
        color: white;
        border-radius: 20px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 70%;
        align-self: flex-end;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .message-container {
        display: flex;
        flex-direction: column;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "welcome_message_displayed" not in st.session_state:
        st.session_state.welcome_message_displayed = False
    if "saved_chats" not in st.session_state:
        st.session_state.saved_chats = []

initialize_session_state()

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Physical activity information
physical_activity_info = """
[Your provided information about physical activity]
"""

# Function to run the model and get a response
def run_model(user_input):
    system_message = """You are an expert in motivational interviewing, skilled at helping clients explore and resolve ambivalence about behaviour change, specifically focused on physical activity. Use open-ended questions, affirmations, reflective listening, and summaries to guide the conversation. Welcome discussion about physical activity and its benefits."""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]
    
    # Add context from physical activity information
    messages.append({"role": "system", "content": f"Here's some relevant information about physical activity: {physical_activity_info}"})
    
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:virtual-health-labs-ltd:physicalactivity:9oqjIysX",
        messages=messages
    )
    
    return response.choices[0].message.content

def summarize_conversation():
    chat_log = " ".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
    summary_prompt = f"Please summarize the following chat log about physical activity:\n{chat_log}"
    
    summary = run_model(summary_prompt)
    
    st.session_state.chat_history.append({"role": "assistant", "content": f"Summary: {summary}"})

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

def reset_chat():
    st.session_state.chat_history = []
    st.session_state.welcome_message_displayed = False
    st.experimental_rerun()

def save_chat():
    chat_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "chat_history": st.session_state.chat_history
    }
    st.session_state.saved_chats.append(chat_data)
    st.success("Chat history saved")

def get_saved_chats():
    return st.session_state.saved_chats

def load_chat(chat_data):
    st.session_state.chat_history = chat_data['chat_history']
    st.session_state.welcome_message_displayed = True
    st.experimental_rerun()

def export_chat():
    chat_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history])
    b64 = base64.b64encode(chat_text.encode()).decode()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_export_{timestamp}.txt"
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Chat History</a>'
    return href

def show_info():
    st.markdown("""
    <div style="padding: 10px; border-radius: 5px; background-color: #007bff; color: white;">
    <p>The VHL Make-a-change Coachbot is written by Gary Stanyard of Virtual Health Labs.</p>
    <p>You can find out more about VHL here: <a href="https://strategichealth.kartra.com/page/Coachbot" target="_blank" style="color: white;">VHL</a></p>
    </div>
    """, unsafe_allow_html=True)

welcome_messages = [
    "Hi there! I'm a coach specializing in motivational interviewing for physical activity. What change are you considering in your activity levels?",
    "Hello! I'm here to guide you through the process of becoming more physically active. What would you like to focus on today?",
    "Welcome! As a motivational interviewing coach for physical activity, I'm here to support you. What changes in your physical activity are you thinking about making?"
]

def main():
    st.title("✨VHL Physical Activity Coachbot")

    if "welcome_subheader" not in st.session_state:
        st.session_state.welcome_subheader = random.choice(welcome_messages)

    if 'show_info' not in st.session_state:
        st.session_state.show_info = False

    if st.button("ℹ️ About", help="Click to toggle information"):
        st.session_state.show_info = not st.session_state.show_info

    if st.session_state.show_info:
        show_info()

    chat_container = st.container()
    input_container = st.container()
    controls_container = st.container()

    with chat_container:
        st.subheader(st.session_state.welcome_subheader)
        
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if message['role'] == 'assistant':
                st.markdown(f'<div class="message-container" style="display: flex; justify-content: flex-end;"><div class="assistant-message">{message["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message-container" style="display: flex; justify-content: flex-start;"><div class="user-message">{message["content"]}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with input_container:
        user_input = st.chat_input("Type your message...", key="user_input")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.spinner("Thinking..."):
                assistant_response = run_model(user_input)

            if assistant_response:
                message_placeholder = st.empty()
                full_response = ""
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(f'<div class="message-container" style="display: flex; justify-content: flex-end;"><div class="assistant-message">{full_response}</div></div>', unsafe_allow_html=True)
                    time.sleep(0.05)
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            st.experimental_rerun()

    with controls_container:
        if st.session_state.get("chat_history"):
            sentiment = analyze_sentiment(" ".join(msg["content"] for msg in st.session_state["chat_history"]))
            st.write(f'Sentiment: {sentiment:.2f}')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Start Over"):
                reset_chat()
        with col2:
            if st.button("Save Chat"):
                save_chat()
        with col3:
            if st.button("Summarize"):
                summarize_conversation()
        with col4:
            st.markdown(export_chat(), unsafe_allow_html=True)

        saved_chats = get_saved_chats()
        if saved_chats:
            selected_chat = st.selectbox(
                "Load a saved chat",
                options=saved_chats,
                format_func=lambda x: x['timestamp']
            )
            if st.button("Load Selected Chat"):
                load_chat(selected_chat)

if __name__ == "__main__":
    main()
