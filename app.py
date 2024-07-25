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
import re
from collections import Counter

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

# Load the change talk data from file
def load_change_talk_data(file_path='changetalk.jsonl'):
    try:
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        st.error(f"Change talk data file not found: {file_path}")
        return []
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON in change talk data file: {file_path}")
        return []

change_talk_data = load_change_talk_data()

# Create a dictionary mapping statements to stages
change_talk_dict = {item['statement'].lower(): item['stage'] for item in change_talk_data if item['statement']}

# Initialize session state
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "welcome_message_displayed" not in st.session_state:
        st.session_state.welcome_message_displayed = False
    if "saved_chats" not in st.session_state:
        st.session_state.saved_chats = []
    if "change_talk_scores" not in st.session_state:
        st.session_state.change_talk_scores = []

initialize_session_state()

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Physical activity information
physical_activity_info = """
[The evidence for increasing physical activity continues to build. It’s the right step!
Regular physical activity is associated with a reduced risk of a range of diseases including some cancers and dementia. There is also evidence that it can help to prevent and manage many common chronic conditions and diseases, many of which are on the rise and affecting people at an earlier age.
Physical activity is as good or better than treatment with drugs for many conditions, such as type 2 diabetes and lower back pain, and has a much lower risk of any harm.

In addition to supporting good physical and mental health and functioning, regular physical activity also contributes to a range of wider social, environmental and economic benefits for individuals, communities and wider society. Addressing physical activity can also benefit a broad range of wider priorities at a local level, such as reducing air pollution and increasing social cohesion. Wider benefits come primarily from physical activities undertaken in community setting such as walking, cycling, active recreation, sport and play. The relevance and importance of these benefits vary according to life stage and other factors. Social prescribing enables individuals presenting through primary health care to be signposted and connected to local organisations, groups and activities. There are social prescribing schemes that focus on physical activity and staff with knowledge of the resources available in the local community to match individuals to opportunities and support them to engage in activities. In some social prescribing schemes, link workers or health trainers and health champions, signpost and support clients to become involved.
 Let's talk about exercise and chronic diseases. It's important to understand the basics before starting any new exercise routine, especially if you have a long-lasting condition like heart disease, diabetes, depression, or joint pain.

First things first, it's always best to consult with your healthcare provider before starting any new exercise routine. They can help you determine which exercises are safe for you and how often to do them.

Now, let's talk about the benefits of exercise for people with chronic diseases. Regular physical activity can help manage symptoms, improve overall health, and even prevent some conditions. For example, exercise can help reduce the risk of developing heart disease, type 2 diabetes, and certain types of cancer.

There are different types of exercises that can benefit people with chronic diseases. Aerobic exercise, such as brisk walking, cycling, or swimming, can help improve cardiovascular health, increase stamina, and control weight. Strength training, such as weightlifting or bodyweight exercises, can help build muscle and improve muscle strength, which can make daily activities easier and reduce the risk of falls. Flexibility exercises, such as stretching, can help improve joint mobility and reduce the risk of injury.

Balance exercises, such as tai chi or standing on one leg, can also be beneficial for people with chronic diseases, especially older adults or those who have trouble moving. These exercises can help improve balance and reduce the risk of falls.

In addition to these exercises, it's important to remember to start slowly and gradually increase the intensity and duration of your workouts. It's also important to listen to your body and take rest days when needed.

Finally, it's important to find exercises that you enjoy and look forward to doing. This will help you stay motivated and make exercise a regular part of your routine.

So, there you have it! Exercise and chronic diseases go hand in hand, and it's important to understand the basics before starting any new exercise routine. With the right guidance and support, anyone can benefit from regular physical activity, regardless of their health status.]
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

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

def analyze_change_talk(text):
    sentences = re.split(r'[.!?]+', text.lower())
    stage_counts = Counter()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence in change_talk_dict:
            stage = change_talk_dict[sentence]
            stage_counts[stage] += 1
        else:
            for statement, stage in change_talk_dict.items():
                if statement in sentence:
                    stage_counts[stage] += 1
                    break
    
    total_statements = sum(stage_counts.values())
    if total_statements == 0:
        return 0, {}
    
    stage_weights = {
        'pre': 0,
        'contemplation': 1,
        'planning': 2,
        'action': 3,
        'maintenance': 4
    }
    
    weighted_sum = sum(stage_weights[stage] * count for stage, count in stage_counts.items())
    change_talk_score = weighted_sum / total_statements
    normalized_score = change_talk_score / max(stage_weights.values())
    stage_percentages = {stage: (count / total_statements) * 100 for stage, count in stage_counts.items()}
    
    return normalized_score, stage_percentages

def reset_chat():
    st.session_state.chat_history = []
    st.session_state.welcome_message_displayed = False
    st.session_state.change_talk_scores = []
    st.experimental_rerun()

def save_chat():
    chat_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "chat_history": st.session_state.chat_history,
        "change_talk_scores": st.session_state.change_talk_scores
    }
    st.session_state.saved_chats.append(chat_data)
    st.success("Chat history saved")

def get_saved_chats():
    return st.session_state.saved_chats

def load_chat(chat_data):
    st.session_state.chat_history = chat_data['chat_history']
    st.session_state.change_talk_scores = chat_data.get('change_talk_scores', [])
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

            # Analyze change talk
            change_talk_score, stage_percentages = analyze_change_talk(user_input)
            st.session_state.change_talk_scores.append(change_talk_score)

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

            # Display change talk score
            if st.session_state.change_talk_scores:
                avg_change_talk_score = sum(st.session_state.change_talk_scores) / len(st.session_state.change_talk_scores)
                st.write(f'Average Change Talk Score: {avg_change_talk_score:.2f}')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Start Over"):
                reset_chat()
        with col2:
            if st.button("Save Chat"):
                save_chat()
        with col3:
            if st.button("Analyze Change Talk"):
                if st.session_state.chat_history:
                    full_conversation = " ".join([msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"])
                    change_talk_score, stage_percentages = analyze_change_talk(full_conversation)
                    st.write(f"Overall Change Talk Score: {change_talk_score:.2f}")
                    st.write("Stage Percentages:")
                    for stage, percentage in stage_percentages.items():
                        st.write(f"{stage.capitalize()}: {percentage:.2f}%")
                else:
                    st.write("No conversation to analyze yet.")
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
