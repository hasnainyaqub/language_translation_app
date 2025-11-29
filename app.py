import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch

# Set page config
st.set_page_config(
    page_title="English to Urdu Translator",
    page_icon="🌐",
    layout="centered"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    """Load the translation model and tokenizer"""
    model_name = "Helsinki-NLP/opus-mt-en-ur"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, tokenizer, model):
    """Translate English text to Urdu"""
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate translation
    with torch.no_grad():
        translated = model.generate(**inputs)
    
    # Decode the translation
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# App UI
st.title("🌐 English to Urdu Translator")
st.markdown("Translate English text to Urdu using AI-powered neural machine translation")

# Load model
with st.spinner("Loading translation model..."):
    try:
        tokenizer, model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("English Input")
    english_text = st.text_area(
        "Enter English text:",
        height=200,
        placeholder="Type your English text here...",
        key="english_input"
    )

with col2:
    st.subheader("Urdu Translation")
    translation_placeholder = st.empty()

# Translate button
if st.button("Translate 🔄", type="primary", use_container_width=True):
    if english_text.strip():
        with st.spinner("Translating..."):
            try:
                translation = translate_text(english_text, tokenizer, model)
                with col2:
                    translation_placeholder.text_area(
                        "Translation:",
                        value=translation,
                        height=200,
                        key="urdu_output"
                    )
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
    else:
        st.warning("Please enter some English text to translate")

# Example texts
st.markdown("---")
st.subheader("Example Translations")
examples = {
    "Hello, how are you?": "سلام، آپ کیسے ہیں؟",
    "I am fine, thank you": "میں ٹھیک ہوں، شکریہ",
    "The weather is very nice today": "آج موسم بہت اچھا ہے",
    "What is your name?": "آپ کا نام کیا ہے؟",
    "Good morning": "صبح بخیر"
}

st.markdown("Click on an example to try it:")
for english, urdu in examples.items():
    if st.button(f"{english} → {urdu}", key=english):
        st.session_state.english_input = english
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Powered by Helsinki-NLP OPUS-MT Model | Built with Streamlit & Transformers</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Installation instructions in sidebar
with st.sidebar:
    st.header("📦 Installation")
    st.code("""
pip install streamlit
pip install transformers
pip install torch
pip install sentencepiece
    """, language="bash")
    
    st.header("🚀 How to Run")
    st.code("""
streamlit run app.py
    """, language="bash")
    
    st.header("ℹ️ About")
    st.info(
        """
        This app uses the Helsinki-NLP OPUS-MT model 
        for English to Urdu translation. The model is 
        based on the Marian NMT framework and trained 
        on parallel corpora.
        """
    )