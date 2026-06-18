# Project 2: College Student FAQ Chatbot (CodeAlpha AI Internship)

An AI-powered College Student FAQ Assistant. It parses a dataset of 50+ college-related questions (admissions, attendance, fees, exams, library, hostels) and matches student inquiries with high accuracy using NLP techniques (TF-IDF Vectorization and Cosine Similarity).

## Features

* **Interactive Chat UI**: Uses Streamlit's native chat elements for a clean, modern chatting experience.
* **Natural Language Preprocessing**: Integrates NLTK's wordnet lemmatization and punctuation/stopword removal.
* **Smart Matching Engine**: Leverages Scikit-Learn TF-IDF vectorization and Cosine Similarity to find the best match.
* **Confidence Scores**: Displays match confidence levels with color-coded tags.
* **Dynamic Follow-up Questions**: Recommends related questions dynamically based on the current context/category.
* **Typing Animation**: Simulates typing delays for an organic conversational feel.
* **Sidebar Analytics**: Provides metrics about categories and questions loaded in real time.

---

## Folder Structure

```
project2_faq_chatbot/
├── app.py                     # Main chatbot application
├── faq_data.json              # Dataset containing 50+ college Q&As
├── requirements.txt           # Python dependencies
├── README.md                  # Project instructions & details
└── docs/
    └── report.md              # Project report, Viva QA, LinkedIn/Resume templates
```

---

## Technical Stack

* **Language**: Python 3.8+
* **Framework**: Streamlit
* **Libraries**: NLTK, Scikit-Learn (scikit-learn), NumPy, Pandas

---

## Installation & Setup

### 1. Navigate to the Directory
```bash
cd project2_faq_chatbot
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.
