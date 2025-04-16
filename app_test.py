import http.client
import json
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
import re
import asyncio
import nest_asyncio
import warnings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize

warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# Load environment variables
load_dotenv()

# Set tokenizers parallelism to false for compatibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

def init_async():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Initialize GROQ model
def init_groq_model():
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return ChatGroq(
        groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.2
    )

llm_groq = init_groq_model()

# Load tokenizer and model once
@st.cache_resource
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", use_auth_token=True)
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", use_auth_token=True)
    return tokenizer, model

tokenizer, model = load_embedding_model()

# Compute sentence embeddings
def compute_embeddings(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)

    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    return normalize(embeddings, p=2, dim=1)

# Extract PDF text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

# Split text
def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=3000, chunk_overlap=200, length_function=len)
    return splitter.split_text(text)

# Create vector store
def get_vectorstore(text_chunks):
    embeddings = compute_embeddings(text_chunks)
    return FAISS.from_embeddings(embeddings.tolist(), text_chunks)

# Conversation chain
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    return ConversationalRetrievalChain.from_llm(llm=llm_groq, retriever=vectorstore.as_retriever(), memory=memory, return_source_documents=True)

# Extract job features
def extract_job_features(text):
    skills = re.findall(r'\b(Java|Python|Data Science|Machine Learning|Deep Learning|Software Engineer|Data Engineer|AI|NLP|C\+\+|SQL|TensorFlow|Keras)\b', text, re.IGNORECASE)
    titles = re.findall(r'\b(Engineer|Data Scientist|Developer|Manager|Analyst|Consultant)\b', text, re.IGNORECASE)
    return list(set(skills + titles)) or ["General"]

# Fetch job recommendations
def get_job_recommendations(features):
    host = "jooble.org"
    api_key = os.getenv("JOOBLE_API_KEY")
    connection = http.client.HTTPConnection(host)
    headers = {"Content-type": "application/json"}
    body = json.dumps({"keywords": ", ".join(features), "location": "Remote"})

    try:
        connection.request("POST", f"/api/{api_key}", body, headers)
        response = connection.getresponse()
        data = response.read()
        jobs = json.loads(data).get("jobs", [])
        return [{"title": j.get("title", "Job Title"), "company": j.get("company", "Company"), "link": j.get("link", "#"), "description": clean_job_description(j.get("snippet", "No description."))} for j in jobs]
    except Exception as e:
        st.error(f"Error fetching job data: {e}")
        return []

def clean_job_description(desc):
    desc = re.sub(r'&nbsp;|&#39;|<[^>]+>', '', desc)
    highlight = re.findall(r'\b(?:Python|Java|TensorFlow|Keras|Machine Learning|AI|NLP|Deep Learning|Engineer|Data Scientist|Developer|Analyst)\b', desc, re.IGNORECASE)
    for word in highlight:
        desc = re.sub(rf"\b{re.escape(word)}\b", f"**{word}**", desc)
    return desc

def handle_userinput(question):
    if question:
        try:
            response = st.session_state.conversation.invoke({"question": question})
            st.write(response.get('answer'))
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")

def main():
    st.set_page_config(page_title="Job Assistant Chatbot", page_icon=":briefcase:")
    st.header("Job Assistant Chatbot :briefcase:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    try:
        init_async()

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "job_recommendations" not in st.session_state:
            st.session_state.job_recommendations = []

        tab = st.sidebar.radio("Choose a tab", ["Chatbot", "Job Recommendations"])

        if tab == "Chatbot":
            user_question = st.text_input("Ask a question about your Resume:")
            if user_question:
                handle_userinput(user_question)

            st.sidebar.subheader("Your documents")
            pdf_docs = st.sidebar.file_uploader("Upload resumes (PDF)", accept_multiple_files=True)
            if st.sidebar.button("Process"):
                if pdf_docs:
                    with st.spinner("Processing..."):
                        try:
                            text = get_pdf_text(pdf_docs)
                            chunks = get_text_chunks(text)
                            vectorstore = get_vectorstore(chunks)
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            features = extract_job_features(text)
                            st.session_state.job_recommendations = get_job_recommendations(features)
                            st.success("Documents processed & jobs updated.")
                        except Exception as e:
                            st.error(f"Error processing documents: {e}")
                else:
                    st.warning("Please upload PDFs first.")

        elif tab == "Job Recommendations":
            st.header("Recommended Jobs 💼")
            if st.session_state.job_recommendations:
                for job in st.session_state.job_recommendations:
                    st.markdown(f"**[{job['title']}]({job['link']})** at **{job['company']}**")
                    st.markdown(f"**Description:** {job['description']}", unsafe_allow_html=True)
            else:
                st.info("Upload your resume under Chatbot tab to get recommendations.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh and try again.")

if __name__ == '__main__':
    main()
