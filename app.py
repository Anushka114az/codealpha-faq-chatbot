import streamlit as st
import json
import os
import time
import string
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Campus Assistant - CodeAlpha FAQ Bot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK resources dynamically
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except Exception as e:
        return False

nltk_status = download_nltk_resources()

# Custom CSS for a professional chatbot look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Title Styling */
    .title-gradient {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F8F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #a0aec0;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metrics panel style */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FF4B4B;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
    }
    
    /* Custom chip badge */
    .confidence-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .badge-high {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    .badge-med {
        background-color: rgba(241, 196, 15, 0.15);
        color: #f1c40f;
        border: 1px solid rgba(241, 196, 15, 0.3);
    }
    .badge-low {
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border: 1px solid rgba(231, 76, 60, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load FAQ dataset
@st.cache_data
def load_faq_data():
    file_path = os.path.join(os.path.dirname(__file__), 'faq_data.json')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Fallback dataset if file is missing
        return [
            {"category": "General", "question": "What is the college address?", "answer": "The college is located at 123 Education Boulevard, Academic City."},
            {"category": "Library", "question": "What are library timings?", "answer": "The library is open from 8:00 AM to 10:00 PM on weekdays."}
        ]

faq_dataset = load_faq_data()

# Text preprocessing function
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

def clean_and_preprocess(text):
    text = text.lower()
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Tokenize
    tokens = text.split()
    # Remove stopwords and lemmatize
    processed_tokens = []
    for token in tokens:
        if token not in stop_words:
            try:
                lemma = lemmatizer.lemmatize(token)
            except:
                lemma = token
            processed_tokens.append(lemma)
    return " ".join(processed_tokens)

# Prepare corpus and vectorizer
@st.cache_resource
def init_nlp_model(faq_data):
    questions = [item['question'] for item in faq_data]
    preprocessed_questions = [clean_and_preprocess(q) for q in questions]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)
    
    return vectorizer, tfidf_matrix, questions

vectorizer, tfidf_matrix, raw_questions = init_nlp_model(faq_dataset)

# Function to get matching answer
def get_best_match(user_query, threshold=0.25):
    cleaned_query = clean_and_preprocess(user_query)
    if not cleaned_query.strip():
        return {
            "answer": "Please ask a complete question so I can assist you better.",
            "confidence": 0.0,
            "category": "None",
            "matched_question": "",
            "suggestions": []
        }
        
    query_vector = vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    matched_faq = faq_dataset[best_match_idx]
    
    # If confidence is below threshold, find fallback and category-based suggestions
    if best_score < threshold:
        # Get general suggestions from random categories or top 3 highest anyway
        top_indices = np.argsort(similarities)[-3:][::-1]
        suggestions = [faq_dataset[idx]['question'] for idx in top_indices if similarities[idx] > 0.05]
        
        # If no similarity matches, suggest general popular questions
        if not suggestions:
            suggestions = [
                "What is the minimum attendance requirement to sit for exams?",
                "How and where can I pay my semester fees?",
                "What are the working hours of the central library?"
            ]
            
        return {
            "answer": "I'm sorry, I couldn't find a direct answer to that specific question in my database. Could you please rephrase it, or check one of the related questions below?",
            "confidence": float(best_score),
            "category": "None",
            "matched_question": "",
            "suggestions": suggestions
        }
        
    # Retrieve top 3 recommendations (excluding the exact match itself)
    top_indices = np.argsort(similarities)[-4:][::-1]
    suggestions = []
    for idx in top_indices:
        if idx != best_match_idx and similarities[idx] > 0.1:
            suggestions.append(faq_dataset[idx]['question'])
            
    # If not enough similarity-based suggestions, get questions from the same category
    if len(suggestions) < 3:
        cat_matches = [item['question'] for item in faq_dataset if item['category'] == matched_faq['category'] and item['question'] != matched_faq['question']]
        for q in cat_matches:
            if q not in suggestions:
                suggestions.append(q)
            if len(suggestions) >= 3:
                break
                
    return {
        "answer": matched_faq['answer'],
        "confidence": float(best_score),
        "category": matched_faq['category'],
        "matched_question": matched_faq['question'],
        "suggestions": suggestions[:3]
    }

# Streamlit Layout
# Sidebar with stats and directory view
with st.sidebar:
    st.markdown("<div style='text-align: center;'><h2 style='color:#FF4B4B;'>🏫 College Info Desk</h2></div>", unsafe_allow_html=True)
    st.write("---")
    
    # Statistics cards
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(faq_dataset)}</div><div class='metric-label'>FAQs Available</div></div>", unsafe_allow_html=True)
    with col_b:
        categories = list(set([item['category'] for item in faq_dataset]))
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(categories)}</div><div class='metric-label'>Categories</div></div>", unsafe_allow_html=True)
        
    st.write("### 📂 Categories Covered")
    for cat in sorted(categories):
        count = len([item for item in faq_dataset if item['category'] == cat])
        st.write(f"- **{cat}** ({count} questions)")
        
    st.write("---")
    st.write("### 🛠️ Technology Stack")
    st.caption("**Core Engine**: TF-IDF & Cosine Similarity")
    st.caption("**Preprocessing**: NLTK WordNet Lemmatizer")
    st.caption("**UI Framework**: Streamlit Native Chat")
    
    # Clear chat button
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.suggestions = [
            "What is the minimum attendance requirement to sit for exams?",
            "How and where can I pay my semester fees?",
            "What are the working hours of the central library?",
            "How do I apply for admission to the college?"
        ]
        st.rerun()

# Main Chat interface
st.markdown("<h1 class='title-gradient'>🎓 Campus FAQ Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask about admissions, hostel rules, fees, exams, and library timings.</p>", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "suggestions" not in st.session_state:
    st.session_state.suggestions = [
        "What is the minimum attendance requirement to sit for exams?",
        "How and where can I pay my semester fees?",
        "What are the working hours of the central library?",
        "How do I apply for admission to the college?"
    ]

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "confidence" in message:
            conf = message["confidence"]
            cat = message["category"]
            if conf > 0.0:
                badge_class = "badge-high" if conf > 0.6 else ("badge-med" if conf > 0.4 else "badge-low")
                st.markdown(f"<span class='confidence-badge {badge_class}'>Matched {cat} | Confidence: {conf:.0%}</span>", unsafe_allow_html=True)

# Function to handle processing a question
def process_user_question(question_text):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question_text})
    
    # Show user message immediately
    with st.chat_message("user"):
        st.write(question_text)
        
    # Get response
    result = get_best_match(question_text)
    
    # Display assistant response with simulated typing
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate typing animation
        for chunk in result["answer"].split(" "):
            full_response += chunk + " "
            time.sleep(0.04)
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(result["answer"])
        
        # Show confidence badge if appropriate
        conf = result["confidence"]
        if conf > 0.0:
            badge_class = "badge-high" if conf > 0.6 else ("badge-med" if conf > 0.4 else "badge-low")
            st.markdown(f"<span class='confidence-badge {badge_class}'>Matched {result['category']} | Confidence: {conf:.0%}</span>", unsafe_allow_html=True)
            
    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": result["answer"],
        "confidence": result["confidence"],
        "category": result["category"]
    })
    
    # Save new suggestions to session state
    st.session_state.suggestions = result["suggestions"]
    st.rerun()

# Check for input from st.chat_input
user_input = st.chat_input("Ask a question about the college...")

# Check for button click on suggested questions
clicked_suggestion = None
if st.session_state.suggestions:
    st.write("💡 *Suggested questions you can click:*")
    # Display suggestions as inline buttons
    cols = st.columns(len(st.session_state.suggestions))
    for idx, sug in enumerate(st.session_state.suggestions):
        with cols[idx]:
            if st.button(sug, key=f"sug_btn_{idx}"):
                clicked_suggestion = sug

# If there is a clicked suggestion or text input, trigger processing
if clicked_suggestion:
    process_user_question(clicked_suggestion)
elif user_input:
    process_user_question(user_input)

# Welcome message if chat is empty
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.write("👋 Hello! I am your College Student Assistant. How can I help you today? You can ask me anything about college admissions, hostel guidelines, class attendance, fees, exams, or the library.")
