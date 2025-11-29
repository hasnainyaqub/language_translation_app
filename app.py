import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch

# Set page config
st.set_page_config(
    page_title="Urdu to English Translator",
    page_icon="🌐",
    layout="centered"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    """Load the translation model and tokenizer"""
    model_name = "Helsinki-NLP/opus-mt-ur-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, tokenizer, model):
    """Translate Urdu text to English"""
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate translation
    with torch.no_grad():
        translated = model.generate(**inputs)
    
    # Decode the translation
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# App UI
st.title("🌐 Urdu to English Translator")
st.markdown("Translate Urdu text to English using AI-powered neural machine translation")

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
    st.subheader("Urdu Input")
    urdu_text = st.text_area(
        "Enter Urdu text:",
        height=200,
        placeholder="یہاں اردو متن درج کریں...",
        key="urdu_input"
    )

with col2:
    st.subheader("English Translation")
    translation_placeholder = st.empty()

# Translate button
if st.button("Translate 🔄", type="primary", use_container_width=True):
    if urdu_text.strip():
        with st.spinner("Translating..."):
            try:
                translation = translate_text(urdu_text, tokenizer, model)
                with col2:
                    translation_placeholder.text_area(
                        "Translation:",
                        value=translation,
                        height=200,
                        key="english_output"
                    )
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
    else:
        st.warning("Please enter some Urdu text to translate")

# Example texts
st.markdown("---")
st.subheader("Example Translations")
examples = {
    "سلام، آپ کیسے ہیں؟": "Hello, how are you?",
    "میں ٹھیک ہوں، شکریہ": "I am fine, thank you",
    "آج موسم بہت اچھا ہے": "The weather is very nice today"
}

st.markdown("Click on an example to try it:")
for urdu, english in examples.items():
    if st.button(f"{urdu}", key=urdu):
        st.session_state.urdu_input = urdu
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
        for Urdu to English translation. The model is 
        based on the Marian NMT framework and trained 
        on parallel corpora.
        """
    )