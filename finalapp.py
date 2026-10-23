import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import time
import base64
import io
import streamlit as st
from dotenv import load_dotenv
import pymupdf as fitz
from PIL import Image
from openai import OpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# 1. Environment & API Key Setup
load_dotenv()

api_key = os.getenv("api") or os.getenv("api") or os.getenv("api") or ""
os.environ["api"] = api_key

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "deepseek-ai/deepseek-v4.1-flash"

# Page Configuration
st.set_page_config(page_title="NVIDIA NIM Multi-Engine", layout="wide")
st.title("NVIDIA NIM Multimodal & Chat Suite")

# Sidebar Mode Switcher
mode = st.sidebar.radio(
    "Select Operating Mode:", 
    [
        "Direct Text RAG (DeepSeek Context)", 
        "Direct Visual Analysis (PNG, JPG, PDF)", 
        "General LLM Chat (With Memory)"
    ]
)

# Helper function to downsample and compress images for low payload latency
def process_and_compress_image(image_bytes, max_size=(1024, 1024)):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Initialize LangChain LLM model for Direct Context Analysis
deepseek_llm = ChatNVIDIA(
    base_url=BASE_URL,
    model=MODEL_NAME,
    nvidia_api_key=api_key,
    timeout=180
)

# Native OpenAI Client for Visual Analysis and General Chat
native_client = OpenAI(
    base_url=BASE_URL,
    api_key=api_key,
    timeout=180.0
)

# =====================================================================
# MODE 1: DIRECT TEXT RAG (DEEPSEEK DIRECT CONTEXT)
# =====================================================================
if mode == "Direct Text RAG (DeepSeek Context)":
    st.subheader("Direct Context Document Analysis with DeepSeek Flash")

    uploaded_files = st.file_uploader(
        "Upload PDF documents for analysis:", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        if "extracted_text" not in st.session_state or st.sidebar.button("Re-process Documents"):
            combined_text = ""
            file_names = []
            
            with st.spinner("Extracting text from uploaded PDF(s)..."):
                for uploaded_file in uploaded_files:
                    file_names.append(uploaded_file.name)
                    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    for page in pdf_doc:
                        combined_text += page.get_text() + "\n"
            
            st.session_state.extracted_text = combined_text
            st.session_state.file_names = file_names
            st.success(f"Successfully processed {len(file_names)} file(s) ({len(combined_text)} characters extracted).")

    if "extracted_text" in st.session_state and st.session_state.extracted_text:
        with st.expander("Preview Extracted Document Text"):
            st.text_area("Document Content Overview", st.session_state.extracted_text[:3000] + "...", height=200)

        user_question = st.text_input("Ask a question about the document(s):")

        if user_question:
            with st.spinner("DeepSeek Flash is reading document context and generating response..."):
                start_time = time.process_time()
                
                # Direct context injection into prompt
                prompt_text = f"""
                You are an expert document analysis assistant. Answer the user's question based strictly on the provided context below.
                
                <context>
                {st.session_state.extracted_text[:120000]}
                </context>
                
                Question: {user_question}
                """
                
                try:
                    response = deepseek_llm.invoke(prompt_text)
                    elapsed_time = time.process_time() - start_time

                    st.write(f"**Response from DeepSeek Flash (processed in {elapsed_time:.2f}s):**")
                    st.write(response.content)
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    else:
        st.info("Please upload one or more PDF documents above to begin querying.")

# =====================================================================
# MODE 2: DIRECT VISUAL ANALYSIS (OPTIMIZED NATIVE STREAMING)
# =====================================================================
elif mode == "Direct Visual Analysis (PNG, JPG, PDF)":
    st.subheader("Direct Multimodal Analysis (Images & PDF Visuals)")

    uploaded_file = st.file_uploader(
        "Upload a PNG, JPG image or a PDF document:", 
        type=["png", "jpg", "jpeg", "pdf"]
    )

    if uploaded_file:
        base64_image = None

        if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            image_bytes = uploaded_file.read()
            base64_image = process_and_compress_image(image_bytes)
            st.image(image_bytes, caption="Uploaded Image Preview", width=450)

        elif uploaded_file.type == "application/pdf":
            pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            page = pdf_doc.load_page(0)
            pix = page.get_pixmap(dpi=150)
            image_bytes = pix.tobytes("png")
            base64_image = process_and_compress_image(image_bytes)
            st.image(image_bytes, caption="PDF Page 1 Rendered Image", width=450)

        visual_prompt = st.text_input("Ask a question about this visual/image (Press Enter):")

        if visual_prompt:
            with st.chat_message("assistant"):
                try:
                    completion = native_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": visual_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                    }
                                ]
                            }
                        ],
                        temperature=0.7,
                        max_tokens=2048,
                        stream=True
                    )

                    def get_visual_stream():
                        for chunk in completion:
                            if chunk.choices and chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content

                    st.write_stream(get_visual_stream())

                except Exception as e:
                    st.error(f"Visual analysis failed: {str(e)}")

# =====================================================================
# MODE 3: GENERAL CHAT WITH MEMORY
# =====================================================================
else:
    st.subheader("General Conversational Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message here...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            try:
                recent_messages = st.session_state.chat_history[-6:]
                formatted_messages = [
                    {"role": m["role"], "content": m["content"]} for m in recent_messages
                ]

                completion = native_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=formatted_messages,
                    temperature=0.7,
                    max_tokens=2048,
                    stream=True
                )

                def get_stream():
                    for chunk in completion:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                full_response = st.write_stream(get_stream())
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Chat request failed: {str(e)}")