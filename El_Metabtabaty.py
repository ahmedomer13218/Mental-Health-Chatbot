import streamlit as st
import os
import re
import logging
import time
import speech_recognition as sr  # Import SpeechRecognition for voice input
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import pandas as pd
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load API keys
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(groq_api_key=groq_api_key, model='Llama3-70b-8192')

# Contextualize and Q&A prompts
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = """Answer this question using the provided context only and give the users the response to solve the problem. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \

context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Function to create vector embeddings
def create_vector_embedding():
    if 'vectors' not in st.session_state:
        df = pd.read_csv('data/mentalHealthConversations.csv')
        st.session_state.documents = []
        for _, row in df.iterrows():
            content = row.to_string()  # Convert the row to a string
            doc = Document(page_content=content)
            st.session_state.documents.append(doc)
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectors = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

if 'store' not in st.session_state:
    st.session_state.store = {}

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'context' not in st.session_state:
    st.session_state.context = []

# Function to get chat history by session
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Function to record and transcribe voice input
def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        audio = r.listen(source)
        st.info("Processing...")
        try:
            query = r.recognize_google(audio)
            st.success(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand that. Please try again.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
    return ""

# Streamlit UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Enter your questions")

# Add voice input button
if st.button("Use Voice Input"):
    voice_query = voice_input()
    if voice_query:
        user_prompt = voice_query

with st.sidebar:
    if 'vectors' not in st.session_state:
        if st.button("Document Embedding"):
            create_vector_embedding()
            st.write("Vector Database is ready")
        else:
            st.write('Click the button first to initialize the chat vector store')
    else:
        st.write("Vector Database is loaded")

if user_prompt:
    retriever = st.session_state.vectors.as_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    response = conversational_rag_chain.invoke(
        {"input": user_prompt},
        config={"configurable": {"session_id": "ahmedomar"}},
    )

    st.session_state.chat_history.append(HumanMessage(content=user_prompt))
    st.session_state.chat_history.append(AIMessage(content=response['answer']))

    model_response = response['answer']

    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    def chunk_response(text, chunk_size=20):
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in chunk_response(model_response):
            full_response += chunk
            message_placeholder.markdown(full_response)
            time.sleep(0.1)  # Simulate a delay between chunks to mimic streaming
        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("----------------------------")
                if doc.page_content not in st.session_state.context:
                    st.session_state.context.append(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": model_response})

# Additional code for text cleaning and logging
def clean_text(text):
    cleaned = re.sub(r'\*+\s*', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
