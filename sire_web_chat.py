import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Get OpenAI key from Streamlit secrets
OPENAI_API_KEY = st.secrets["openai_api_key"]

# Set up file and folder
INDEX_FOLDER = "faiss_index"
support_file = "support_chats.txt"
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Load or build vectorstore
if os.path.exists(INDEX_FOLDER):
    vectorstore = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)
else:
    loader = TextLoader(support_file)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(INDEX_FOLDER)

# Custom prompt to keep tone casual and avoid names/templates
custom_prompt = """
You are a support agent for a company called Sire. Use the following past conversations to guide your tone and style.
Never include template text like "your name" or make up random names or account info.
Only respond with relevant, helpful, casual but respectful answers — matching the customer's energy and tone. Don't ask for irrelevant info.

Support History:
{context}

Customer: {question}
Support:"""

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=custom_prompt,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4", temperature=0.3),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
)

# Streamlit app frontend
st.set_page_config(page_title="Sire Support Chat", layout="centered")
st.markdown("<h1 style='text-align: center;'>💬 Sire Support Chat</h1>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    st.markdown(f"<div style='padding: 10px; background-color: #222; color: #fff; border-radius: 8px; margin-bottom: 10px;'><b>You:</b> {entry['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='padding: 10px; background-color: #333; color: #e0e0e0; border-radius: 8px; margin-bottom: 20px;'><b>Bot:</b> {entry['bot']}</div>", unsafe_allow_html=True)

user_input = st.text_input("Type your message...", key="input", label_visibility="collapsed")

if st.button("Send"):
    if user_input.strip():
        response = qa_chain.run(user_input)
        st.session_state.chat_history.append({"user": user_input, "bot": response})
        st.rerun()
