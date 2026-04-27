import os
import warnings

warnings.filterensitive = False
warnings.filterwarnings("ignore", message=".*torch.classes.*")

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from rag import RecipeRetriever

load_dotenv()

st.set_page_config(
    page_title="Recipe Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }
    section[data-testid="stSidebar"] { background: rgba(255,255,255,0.05); backdrop-filter: blur(12px); border-right: 1px solid rgba(255,255,255,0.1); }
    .stChatMessage { border-radius: 16px !important; padding: 4px !important; }
    [data-testid="stChatMessage"][aria-label="user"] { background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(168,85,247,0.25)) !important; border: 1px solid rgba(168,85,247,0.4) !important; }
    [data-testid="stChatMessage"][aria-label="assistant"] { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; }
    .stChatInputContainer { background: rgba(255,255,255,0.05) !important; border-radius: 16px !important; border: 1px solid rgba(255,255,255,0.15) !important; backdrop-filter: blur(8px); }
    .stButton > button { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; }
    .stAlert { border-radius: 12px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Connecting to Gemini...")
def load_gemini():
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key, transport='rest')
        return genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=(
                "You are a friendly and knowledgeable Recipe Assistant. "
                "You help users discover recipes, explain cooking techniques, "
                "suggest substitutions, and answer any food-related questions. "
                "When you have relevant recipes from the knowledge base, summarise "
                "and present them clearly with ingredients and steps. "
                "Always maintain a warm, encouraging tone."
            ),
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
        )
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")
        return None

@st.cache_resource(show_spinner="Loading recipe index...")
def load_retriever():
    try:
        from rag import INDEX_PATH
        if not os.path.exists(INDEX_PATH):
            with st.status("Initializing recipe database for the first time..."):
                import ingest
                ingest.main()
        return RecipeRetriever()
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return None

gemini_model = load_gemini()
retriever = load_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("<h1 style='color: #c4b5fd; font-size:1.6rem;'>Recipe Assistant</h1>", unsafe_allow_html=True)
    st.divider()
    
    llm_ok = gemini_model is not None
    idx_ok = retriever is not None
    
    st.markdown(f"{'ONLINE' if llm_ok else 'OFFLINE'} **Gemini LLM**")
    st.markdown(f"{'ONLINE' if idx_ok else 'OFFLINE'} **Recipe Index** ({'{:,}'.format(retriever._index.ntotal) if idx_ok else '0'} recipes)")
    
    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("<h1 style='text-align:center; color:#c4b5fd;'>Recipe Assistant</h1>", unsafe_allow_html=True)
st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a recipe question...", disabled=(not llm_ok or not idx_ok))

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    standalone_query = user_input
    if len(st.session_state.messages) > 1:
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:-1]])
        rewrite_prompt = (
            f"Given the following conversation and a follow up question, rephrase the follow up question "
            f"to be a standalone search query for a recipe database.\n\n"
            f"Chat History:\n{history_str}\n\n"
            f"Follow Up Input: {user_input}\n\n"
            f"Standalone query:"
        )
        try:
            rewrite_response = gemini_model.generate_content(rewrite_prompt)
            standalone_query = rewrite_response.text.strip()
        except:
            standalone_query = user_input

    filters = RecipeRetriever.parse_query_filters(standalone_query)
    recipes = retriever.retrieve(standalone_query, k=5, ingredient_filter=filters["ingredient_filter"] or None, max_steps=filters["max_steps"])
    
    context = "Knowledge base recipes:\n" + "\n".join([f"- {r.title}: {r.instructions}" for r in recipes]) if recipes else "No matching recipes found."
    
    history_context = "Recent history:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:-1]]) + "\n" if len(st.session_state.messages) > 1 else ""
    
    full_prompt = f"{history_context}\n{context}\n\nUser Question: {user_input}"

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = gemini_model.generate_content(full_prompt)
                if response.candidates and response.candidates[0].content.parts:
                    full_response = response.text
                else:
                    full_response = "The AI was unable to generate a response. Please try rephrasing your question."
            except Exception as e:
                full_response = f"Gemini error: {e}"
        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
