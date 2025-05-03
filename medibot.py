import os 
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
import re

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": "512"
        }
    )
    return llm

def clean_numbered_list(text):
    # Remove patterns like 1. 2. 3. or 1) 2) etc.
    cleaned = re.sub(r"(?m)^\s*\d+[\.\)]\s*", "", text)
    return cleaned.strip()

def main():
    st.title("Medibot: Ask your chatbot!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
    
    st.write("Type a message below to start chatting with Medibot.")
    prompt = st.chat_input("Write your prompt here:")
    
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know. Do not make up an answer.
        Only use the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")
        
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = clean_numbered_list(response["result"])
            source_documents = response["source_documents"]

            # Build source string
            source_info = ""
            for doc in source_documents:
                page = doc.metadata.get("page_label", doc.metadata.get("page", "Unknown"))
                source = doc.metadata.get("source", "Unknown Source").replace("\\", "/")
                source_info += f"\n• Page {page} - {source}"

            result_to_show = f"{result}\n\n---\n**Sources:**{source_info}"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
