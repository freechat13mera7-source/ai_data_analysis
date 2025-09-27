import streamlit as st
import pandas as pd
from pandasai.llm.base import LLM
from pandasai import SmartDataframe
from groq import Groq
import streamlit as st

# ----------------------------
# LLM Class (Groq + PandasAI)
# ----------------------------
class GroqLLM(LLM):
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b", temperature: float = 0.7):
        super().__init__()
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def call(self, instruction, value: str = "", suffix: str = ""):
        prompt = f"""{instruction}
{value}
{suffix}

# If the task requires Python code, output ONLY valid Python code (no explanations).
# If the task requires a direct answer (like list, number, or text), output ONLY that value.
"""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        response = completion.choices[0].message.content.strip()

        # Handle code response
        if response.startswith("```"):
            response = response.strip("`")
            response = response.replace("python", "", 1).strip()
            response = response.replace("return ", "")
            return response

        # Try evaluate if Python literal
        try:
            evaluated = eval(response, {"__builtins__": {}})
            return evaluated
        except Exception:
            return response

    @property
    def type(self) -> str:
        return "groq"


# ----------------------------
# Streamlit Web App
# ----------------------------
st.set_page_config(page_title="AI Data Insights", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
        }
        .main-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 1rem;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        .stTextInput>div>div>input, .stTextArea>div>textarea {
            border-radius: 12px;
            padding: 12px;
        }
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            background: linear-gradient(90deg, #3498db, #2980b9);
            color: white;
            font-weight: 600;
            padding: 10px;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #2980b9, #1f6391);
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("<h1 class='main-title'>📊 AI-Powered Data Insights</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload your data, ask questions, and uncover insights instantly.</p>", unsafe_allow_html=True)

# --- Login System ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔑 Login to Continue")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        if st.button("Login"):
            if username == "admin" and password == "password123":
                st.session_state.authenticated = True
                st.success("✅ Login successful!")
            else:
                st.error("❌ Invalid username or password")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- File Upload ---
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head(10))  # preview first 10 rows

    # Init LLM + SmartDataframe
    #llm = GroqLLM(api_key="your-groq-api-key")
    llm = GroqLLM(api_key=st.secrets["GROQ_API_KEY"])
    sdf = SmartDataframe(df, config={"llm": llm})

    # --- Ask Questions ---
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("💬 Ask Questions About Your Data")
        query = st.text_area("Type your question here:", placeholder="e.g. What is the average intraday volatility?", height=100)

        if st.button("Get Insights"):
            if query.strip() == "":
                st.warning("⚠️ Please enter a question.")
            else:
                with st.spinner("🔎 Analyzing your data..."):
                    try:
                        answer = sdf.chat(query)
                        st.success("✅ Answer:")
                        st.markdown(f"<div style='padding:15px; background:#ecf9f2; border-radius:10px; font-size:1.1rem;'>{answer}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"⚠️ Error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

