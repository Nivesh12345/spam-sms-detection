import streamlit as st
import re
import joblib

# Custom styling with CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding-bottom: 20px;
    }
    .result-spam {
        color: #D32F2F;
        font-size: 1.5rem;
        padding: 20px;
        border-radius: 5px;
        border: 2px solid #D32F2F;
        text-align: center;
    }
    .result-ham {
        color: #388E3C;
        font-size: 1.5rem;
        padding: 20px;
        border-radius: 5px;
        border: 2px solid #388E3C;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Define the same clean_text function from your notebook
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_models()

# Define a function for prediction
def predict_message(message):
    cleaned = clean_text(message)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)
    return "Spam" if prediction[0] == 1 else "Ham"

# App layout
st.markdown('<h1 class="main-header">📱 SMS Spam Detector</h1>', unsafe_allow_html=True)

# Add description in an expander
with st.expander("ℹ️ About this app"):
    st.write("""
    This app uses machine learning to detect whether an SMS message is spam or not (ham).
    - **Spam**: Unwanted, fraudulent, or promotional messages
    - **Ham**: Legitimate, genuine messages
    
    Simply enter your message below and click 'Analyze Message' to check!
    """)

# Create a text input with placeholder
user_input = st.text_area(
    "Enter your message:",
    placeholder="Type or paste your SMS message here...",
    height=100
)

# Center the button using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Message", type="primary")

# Show results
if analyze_button:
    if user_input:
        result = predict_message(user_input)
        
        # Show message statistics
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Message Length", len(user_input))
        with col2:
            st.metric("Word Count", len(user_input.split()))
            
        # Show prediction result with custom styling
        st.write("### 📊 Analysis Result:")
        if result == "Spam":
            st.markdown(f'<div class="result-spam">🚫 This message appears to be SPAM</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-ham">✅ This message appears to be legitimate (HAM)</div>', unsafe_allow_html=True)
            
    else:
        st.error("Please enter a message to analyze.")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Made with ❤️ by Nivesh Kumar <br>
    Using Streamlit and Machine Learning
</div>
""", unsafe_allow_html=True)
