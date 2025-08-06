import streamlit as st
import requests
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
API_KEY = "AIzaSyDcO0w9yX0ZG9kE7vwHoa7Tj5tkvLHfjGI"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# --- Streamlit Setup ---
st.set_page_config(page_title="Smart Internship Chatbot", layout="centered")
st.title("🤖 Internship Chatbot: Jobs | Courses | Events")
st.markdown("Ask about jobs, courses, events — or anything else. I'm here to help!")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Synthetic Data ---
SYNTHETIC_DATA = {
    "jobs": {
        "ai": {
            "field": "Artificial Intelligence",
            "skills": ["Python", "TensorFlow", "Machine Learning"],
            "experience": "1–3 years",
            "salary": "₹6–10 LPA"
        },
        "iot": {
            "field": "Internet of Things",
            "skills": ["C", "Microcontrollers", "MQTT", "Edge computing"],
            "experience": "0–2 years",
            "salary": "₹4–7 LPA"
        }
    },
    "courses": {
        "embedded systems": {
            "subjects": ["C programming", "ARM Cortex", "RTOS", "Sensors"],
            "duration": "12 weeks",
            "fee": "₹8,000"
        },
        "ai": {
            "subjects": ["ML", "Deep Learning", "Neural Nets"],
            "duration": "10 weeks",
            "fee": "₹10,000"
        }
    },
    "events": {
        "tech conference": {
            "date": "15th August 2025",
            "location": "Bengaluru",
            "schedule": "Keynote, Talks, Networking"
        },
        "hackathon": {
            "date": "22nd August 2025",
            "location": "Online",
            "schedule": "48hr coding challenge + demos"
        }
    }
}

# --- Category Detection ---
def detect_category(query: str) -> str:
    query = query.lower()
    if any(k in query for k in ["job", "career", "hiring", "salary"]):
        return "job"
    elif any(k in query for k in ["course", "class", "learn", "fee", "subject", "duration"]):
        return "course"
    elif any(k in query for k in ["event", "hackathon", "conference", "seminar", "meetup"]):
        return "event"
    return "general"

# --- Synthetic Data Handler ---
def respond_from_synthetic(query: str, category: str) -> str | None:
    q = query.lower()

    if category == "job":
        for key in SYNTHETIC_DATA["jobs"]:
            if key in q:
                data = SYNTHETIC_DATA["jobs"][key]
                return f"**{data['field']} Job Info**\n\n- Skills: {', '.join(data['skills'])}\n- Experience: {data['experience']}\n- Salary: {data['salary']}"

    elif category == "course":
        for key in SYNTHETIC_DATA["courses"]:
            if key in q:
                data = SYNTHETIC_DATA["courses"][key]
                return f"**{key.title()} Course Details**\n\n- Subjects: {', '.join(data['subjects'])}\n- Duration: {data['duration']}\n- Fee: {data['fee']}"

    elif category == "event":
        for key in SYNTHETIC_DATA["events"]:
            if key in q:
                data = SYNTHETIC_DATA["events"][key]
                return f"**{key.title()}**\n\n- Date: {data['date']}\n- Location: {data['location']}\n- Schedule: {data['schedule']}"

    return None  # fallback to LLM

# --- Prompt Templates ---
PROMPTS = {
    "job": ChatPromptTemplate.from_messages([
        ("system", "You are a job assistant. Provide job field, required skills, experience, and salary."),
        ("user", "{question}")
    ]),
    "course": ChatPromptTemplate.from_messages([
        ("system", "You are a course assistant. Include subjects covered, duration, and fee."),
        ("user", "{question}")
    ]),
    "event": ChatPromptTemplate.from_messages([
        ("system", "You are an event assistant. Include event name, date, location, and schedule."),
        ("user", "{question}")
    ]),
    "general": ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for any kind of question."),
        ("user", "{question}")
    ])
}

# --- Gemini API Call ---
def call_llm_api(prompt_text: str) -> str:
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}]
    }
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        parts = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return parts[0].get("text", "⚠️ No response found.") if parts else "⚠️ Empty response."
    except Exception as e:
        return f"🚨 Error: {e}"

# --- Run Chain ---
def run_chain(query: str):
    category = detect_category(query)
    
    # Check synthetic data first
    synthetic_response = respond_from_synthetic(query, category)
    if synthetic_response:
        return synthetic_response

    # Else, call Gemini through LangChain
    prompt = PROMPTS[category]
    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | (lambda x: call_llm_api(x.to_messages()[1].content))
        | StrOutputParser()
    )
    return chain.invoke(query)

# --- Show Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input Handler ---
if user_input := st.chat_input("Ask something about jobs, courses, or events..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_chain(user_input)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
