import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch

# Set page config
st.set_page_config(
    page_title="Multi-Language Translator",
    page_icon="🌐",
    layout="wide"
)

# Language configurations with top 30 languages
LANGUAGES = {
    "🇵🇰 Urdu": {"model": "Helsinki-NLP/opus-mt-en-ur", "code": "ur"},
    "🇫🇷 French": {"model": "Helsinki-NLP/opus-mt-en-fr", "code": "fr"},
    "🇪🇸 Spanish": {"model": "Helsinki-NLP/opus-mt-en-es", "code": "es"},
    "🇩🇪 German": {"model": "Helsinki-NLP/opus-mt-en-de", "code": "de"},
    "🇮🇹 Italian": {"model": "Helsinki-NLP/opus-mt-en-it", "code": "it"},
    "🇵🇹 Portuguese": {"model": "Helsinki-NLP/opus-mt-en-pt", "code": "pt"},
    "🇷🇺 Russian": {"model": "Helsinki-NLP/opus-mt-en-ru", "code": "ru"},
    "🇨🇳 Chinese": {"model": "Helsinki-NLP/opus-mt-en-zh", "code": "zh"},
    "🇯🇵 Japanese": {"model": "Helsinki-NLP/opus-mt-en-jap", "code": "ja"},
    "🇰🇷 Korean": {"model": "Helsinki-NLP/opus-mt-en-ko", "code": "ko"},
    "🇸🇦 Arabic": {"model": "Helsinki-NLP/opus-mt-en-ar", "code": "ar"},
    "🇮🇳 Hindi": {"model": "Helsinki-NLP/opus-mt-en-hi", "code": "hi"},
    "🇹🇷 Turkish": {"model": "Helsinki-NLP/opus-mt-en-tr", "code": "tr"},
    "🇳🇱 Dutch": {"model": "Helsinki-NLP/opus-mt-en-nl", "code": "nl"},
    "🇵🇱 Polish": {"model": "Helsinki-NLP/opus-mt-en-pl", "code": "pl"},
    "🇸🇪 Swedish": {"model": "Helsinki-NLP/opus-mt-en-sv", "code": "sv"},
    "🇬🇷 Greek": {"model": "Helsinki-NLP/opus-mt-en-el", "code": "el"},
    "🇨🇿 Czech": {"model": "Helsinki-NLP/opus-mt-en-cs", "code": "cs"},
    "🇷🇴 Romanian": {"model": "Helsinki-NLP/opus-mt-en-ro", "code": "ro"},
    "🇭🇺 Hungarian": {"model": "Helsinki-NLP/opus-mt-en-hu", "code": "hu"},
    "🇫🇮 Finnish": {"model": "Helsinki-NLP/opus-mt-en-fi", "code": "fi"},
    "🇩🇰 Danish": {"model": "Helsinki-NLP/opus-mt-en-da", "code": "da"},
    "🇳🇴 Norwegian": {"model": "Helsinki-NLP/opus-mt-en-no", "code": "no"},
    "🇺🇦 Ukrainian": {"model": "Helsinki-NLP/opus-mt-en-uk", "code": "uk"},
    "🇮🇩 Indonesian": {"model": "Helsinki-NLP/opus-mt-en-id", "code": "id"},
    "🇻🇳 Vietnamese": {"model": "Helsinki-NLP/opus-mt-en-vi", "code": "vi"},
    "🇹🇭 Thai": {"model": "Helsinki-NLP/opus-mt-en-th", "code": "th"},
    "🇮🇷 Persian": {"model": "Helsinki-NLP/opus-mt-en-fa", "code": "fa"},
    "🇮🇱 Hebrew": {"model": "Helsinki-NLP/opus-mt-en-he", "code": "he"},
    "🇧🇩 Bengali": {"model": "Helsinki-NLP/opus-mt-en-bn", "code": "bn"}
}

# Example sentences for translation
EXAMPLE_SENTENCES = [
    "Hello, how are you?",
    "Thank you very much",
    "What is your name?",
    "Good morning",
    "I am learning a new language"
]

# Cache the model loading
@st.cache_resource
def load_model(model_name):
    """Load the translation model and tokenizer"""
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

def translate_text(text, tokenizer, model):
    """Translate text using the loaded model"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Initialize session state
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "🇵🇰 Urdu"

# App Title
st.title("🌐 Multi-Language Translator")
st.markdown("**Translate English to 30+ languages using AI-powered neural machine translation**")
st.markdown("---")

# Language Selection Buttons
st.subheader("Select Target Language:")

# Create rows of language buttons (6 buttons per row)
languages_list = list(LANGUAGES.keys())
cols_per_row = 6
num_rows = (len(languages_list) + cols_per_row - 1) // cols_per_row

for row in range(num_rows):
    cols = st.columns(cols_per_row)
    for col_idx in range(cols_per_row):
        lang_idx = row * cols_per_row + col_idx
        if lang_idx < len(languages_list):
            lang = languages_list[lang_idx]
            with cols[col_idx]:
                if st.button(
                    lang, 
                    key=f"lang_{lang}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_language == lang else "secondary"
                ):
                    st.session_state.selected_language = lang
                    st.rerun()

st.markdown("---")

# Get selected language config
selected_lang = st.session_state.selected_language
lang_config = LANGUAGES[selected_lang]

# Display current selection
st.info(f"**Currently translating:** English → {selected_lang}")

# Load model
with st.spinner(f"Loading {selected_lang} translation model..."):
    tokenizer, model, error = load_model(lang_config["model"])
    if error:
        st.error(f"Error loading model: {error}")
        st.warning("Some models may not be available. Please try another language.")
        st.stop()
    else:
        st.success(f"✓ {selected_lang} model loaded successfully!")

# Translation Interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 English Input")
    input_text = st.text_area(
        "Enter English text:",
        height=250,
        placeholder="Type your English text here...",
        key="input_text_area"
    )

with col2:
    st.subheader(f"🌍 {selected_lang} Translation")
    output_placeholder = st.empty()

# Translate button
if st.button("🔄 Translate", type="primary", use_container_width=True):
    if input_text.strip():
        with st.spinner("Translating..."):
            try:
                translation = translate_text(input_text, tokenizer, model)
                with col2:
                    output_placeholder.text_area(
                        "Translation:",
                        value=translation,
                        height=250,
                        key="output_text_area"
                    )
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
    else:
        st.warning("Please enter some English text to translate")

# Example Translations
st.markdown("---")
st.subheader("💡 Try These Examples")
st.markdown("Click on any example to translate it:")

cols = st.columns(5)
for idx, example in enumerate(EXAMPLE_SENTENCES):
    with cols[idx % 5]:
        if st.button(example, key=f"example_{idx}", use_container_width=True):
            with st.spinner("Translating..."):
                try:
                    translation = translate_text(example, tokenizer, model)
                    st.success("✓ Translated!")
                    st.info(f"**English:** {example}\n\n**{selected_lang}:** {translation}")
                except Exception as e:
                    st.error(f"Translation error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p style='color: #666;'>Powered by Helsinki-NLP OPUS-MT Models | Built with Streamlit & Transformers</p>
        <p style='color: #888; font-size: 0.9em;'>Supporting 30+ languages for English translation</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("📦 Installation")
    st.code("""pip install -r requirements.txt""", language="bash")
    
    st.header("🚀 How to Run")
    st.code("""streamlit run app.py""", language="bash")
    
    st.header("ℹ️ About")
    st.info(
        """
        This app uses Helsinki-NLP OPUS-MT models 
        for translation. Each model is downloaded 
        on first use and cached for future sessions.
        
        **Note:** First translation for each language 
        may take a moment while the model loads.
        """
    )
    
    st.header("🌍 Supported Languages")
    st.markdown(f"**Total Languages:** {len(LANGUAGES)}")
    st.markdown("All translations are from English to the selected target language.")