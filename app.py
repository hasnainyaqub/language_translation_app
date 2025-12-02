import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch

# Set page config
st.set_page_config(
    page_title="AI Language Translator | Hasnain Yaqoob",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom container styling */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Language button styling */
    .stButton button {
        border-radius: 12px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        font-weight: 600;
        height: 45px;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Info box styling */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid #667eea;
    }
    
    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        text-align: center;
        box-shadow: 0 -5px 20px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        z-index: 999;
    }
    
    .developer-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .social-links a {
        color: #764ba2;
        text-decoration: none;
        margin: 0 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .social-links a:hover {
        color: #667eea;
        text-decoration: underline;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg .element-container, [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    /* Section divider */
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 2rem 0;
        border-radius: 2px;
    }
    
    /* Card effect for columns */
    .css-1r6slb0 {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Success message styling */
    .stSuccess {
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Language configurations
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
    "🇧🇩 Bengali": {"model": "Helsinki-NLP/opus-mt-en-bn", "code": "bn"}
}

EXAMPLE_SENTENCES = [
    "Hello, how are you?",
    "Thank you very much",
    "What is your name?",
    "Good morning",
    "I am learning a new language"
]

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

# Header
st.markdown("<h1 class='main-header'>🌐 AI Language Translator</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Translate English to 30+ languages using cutting-edge AI technology</p>", unsafe_allow_html=True)

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# Language Selection
st.markdown("### 🎯 Select Your Target Language")
st.markdown("")

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

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# Get selected language
selected_lang = st.session_state.selected_language
lang_config = LANGUAGES[selected_lang]

# Display selection
st.info(f"🎯 **Currently translating:** English → {selected_lang}")

# Load model
with st.spinner(f"🔄 Loading {selected_lang} translation model..."):
    tokenizer, model, error = load_model(lang_config["model"])
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.warning("⚠️ Some models may not be available. Please try another language.")
        st.stop()
    else:
        st.success(f"✅ {selected_lang} model loaded successfully!")

st.markdown("")

# Translation Interface
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📝 English Input")
    input_text = st.text_area(
        "",
        height=250,
        placeholder="Type your English text here and watch the magic happen...",
        key="input_text_area",
        label_visibility="collapsed"
    )

with col2:
    st.markdown(f"### 🌍 {selected_lang} Translation")
    output_placeholder = st.empty()

# Translate button
st.markdown("")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    if st.button("🚀 Translate Now", type="primary", use_container_width=True):
        if input_text.strip():
            with st.spinner("✨ Translating your text..."):
                try:
                    translation = translate_text(input_text, tokenizer, model)
                    with col2:
                        output_placeholder.text_area(
                            "",
                            value=translation,
                            height=250,
                            key="output_text_area",
                            label_visibility="collapsed"
                        )
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Translation error: {str(e)}")
        else:
            st.warning("⚠️ Please enter some English text to translate")

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# Examples
st.markdown("### 💡 Quick Examples - Click to Translate")
st.markdown("")

cols = st.columns(5)
for idx, example in enumerate(EXAMPLE_SENTENCES):
    with cols[idx % 5]:
        if st.button(f"📌 {example}", key=f"example_{idx}", use_container_width=True):
            with st.spinner("✨ Translating..."):
                try:
                    translation = translate_text(example, tokenizer, model)
                    st.success("✅ Translation Complete!")
                    st.info(f"**English:** {example}\n\n**{selected_lang}:** {translation}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer with developer info
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
    <div class='footer'>
        <div class='developer-name'>
            💻 Developed by Hasnain Yaqoob
        </div>
        <div class='social-links'>
            <a href='https://www.linkedin.com/in/hasnainyaqoob' target='_blank'>🔗 LinkedIn</a> |
            <a href='https://x.com/Hasnain_Yaqoob_' target='_blank'>🐦 X (Twitter)</a> |
            <a href='https://github.com/hasnainyaqub' target='_blank'>💻 GitHub</a> |
            <a href='https://www.kaggle.com/hasnainyaqooob' target='_blank'>📊 Kaggle</a>
        </div>
        <div style='margin-top: 0.5rem; color: #888; font-size: 0.9rem;'>
            Powered by Helsinki-NLP OPUS-MT Models | Built with ❤️ using Streamlit & Transformers
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📦 Installation")
    st.code("pip install -r requirements.txt", language="bash")
    
    st.markdown("## 🚀 Quick Start")
    st.code("streamlit run app.py", language="bash")
    
    st.markdown("## 🌍 Features")
    st.markdown(f"""
    ✅ **{len(LANGUAGES)} Languages** Supported  
    ✅ AI-Powered Translation  
    ✅ Real-time Processing  
    ✅ Modern & Intuitive UI  
    ✅ Fast & Accurate Results  
    """)
    
    st.markdown("## ℹ️ About")
    st.info("""
        This translator uses state-of-the-art 
        Helsinki-NLP OPUS-MT models powered by 
        the Marian NMT framework.
        
        **First-time use:** Models are downloaded 
        and cached automatically for each language.
    """)
    
    st.markdown("## 📧 Contact")
    st.markdown("""
        **Hasnain Yaqoob**  
        AI/ML Developer
        
        Feel free to connect on social media!
    """)