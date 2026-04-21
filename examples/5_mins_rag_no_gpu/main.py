# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Minimal RAG (Retrieval-Augmented Generation) pipeline using NVIDIA AI Foundation models.
# The pipeline has 4 main components:
#   1. Document Upload     — users upload files that become the knowledge base
#   2. Models             — an embedding model to vectorize text + an LLM to generate answers
#   3. Vector Store       — a FAISS index that stores and retrieves document chunks by similarity
#   4. Chat Interface     — the user asks questions; relevant chunks are injected into the LLM prompt

import streamlit as st
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
import pickle
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(layout="wide")

# ---------------------------------------------------------------------------
# Component #1 - Document Upload
# ---------------------------------------------------------------------------
# Files uploaded through the sidebar are saved to a local folder (uploaded_docs/).
# On every Streamlit re-run the DirectoryLoader (Component #3) reads from that folder,
# so newly uploaded files are automatically picked up.
with st.sidebar:
    DOCS_DIR = os.path.abspath("./uploaded_docs")
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    st.subheader("Add to the Knowledge Base")
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", accept_multiple_files=True)
        submitted = st.form_submit_button("Upload!")

    if uploaded_files and submitted:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.read())

# ---------------------------------------------------------------------------
# Component #2 - Embedding Model and LLM
# ---------------------------------------------------------------------------
# Both models are served remotely via the NVIDIA API Catalog — no local GPU needed.
#
# - ChatNVIDIA: the LLM used to generate answers. Reads NVIDIA_API_KEY from the environment.
# - NVIDIAEmbeddings: converts text chunks into numeric vectors so they can be compared
#   by similarity. model_type="passage" optimises the embedding for document indexing
#   (vs. "query" which is used at retrieval time for the user's question).
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
document_embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", model_type="passage")

# ---------------------------------------------------------------------------
# Component #3 - Vector Database Store
# ---------------------------------------------------------------------------
# FAISS (Facebook AI Similarity Search) is an in-memory vector index.
# Rather than rebuilding it on every page reload, the index is serialised to
# vectorstore.pkl so it can be reloaded instantly on subsequent runs.
#
# Flow:
#   a) If vectorstore.pkl exists and the user selects "Yes" → load from disk.
#   b) Otherwise, load raw documents from uploaded_docs/, split them into chunks,
#      embed each chunk, build the FAISS index, then save it to disk.
#
# Chunking parameters:
#   chunk_size=512    — each chunk is at most 512 characters. Smaller chunks give
#                       more precise retrieval; larger chunks provide more context.
#   chunk_overlap=200 — consecutive chunks share 200 characters so that sentences
#                       that fall on a boundary are not lost.
with st.sidebar:
    use_existing_vector_store = st.radio("Use existing vector store if available", ["Yes", "No"], horizontal=True)

vector_store_path = "vectorstore.pkl"
# Load every document found in the upload folder (supports txt, pdf, docx, …)
raw_documents = DirectoryLoader(DOCS_DIR).load()

vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None
if use_existing_vector_store == "Yes" and vector_store_exists:
    # Deserialise the previously built FAISS index from disk
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    with st.sidebar:
        st.success("Existing vector store loaded successfully.")
else:
    with st.sidebar:
        if raw_documents and use_existing_vector_store == "Yes":
            with st.spinner("Splitting documents into chunks..."):
                text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=200)
                documents = text_splitter.split_documents(raw_documents)

            with st.spinner("Adding document chunks to vector database..."):
                # Embed every chunk and insert into the FAISS index
                vectorstore = FAISS.from_documents(documents, document_embedder)

            with st.spinner("Saving vector store"):
                with open(vector_store_path, "wb") as f:
                    pickle.dump(vectorstore, f)
            st.success("Vector store created and saved.")
        else:
            st.warning("No documents available to process!", icon="⚠️")

# ---------------------------------------------------------------------------
# Component #4 - LLM Response Generation and Chat
# ---------------------------------------------------------------------------
# The chat follows the RAG pattern:
#   1. Retrieve — find the top-k document chunks most similar to the user's question.
#   2. Augment  — prepend those chunks as "Context:" in the prompt so the LLM can
#                 ground its answer in the uploaded documents.
#   3. Generate — stream the LLM response token-by-token into the chat UI.
#
# If no vector store is available the pipeline falls back to the LLM's general knowledge.
#
# LangChain Expression Language (LCEL) chain:
#   prompt_template → llm → StrOutputParser
#   The pipe operator (|) composes these into a single streamable runnable.
st.subheader("Chat with your AI Assistant, Envie!")

# Persist the full conversation across Streamlit re-runs using session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant named Envie. If provided with context, use it to inform your responses. If no context is available, use your general knowledge to provide a helpful response."),
    ("human", "{input}")
])

chain = prompt_template | llm | StrOutputParser()

user_input = st.chat_input("Can you tell me what NVIDIA is known for?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if vectorstore is not None and use_existing_vector_store == "Yes":
            # Retrieve the most relevant document chunks for the question
            retriever = vectorstore.as_retriever()
            docs = retriever.invoke(user_input)
            # Concatenate chunks into a single context block
            context = "\n\n".join([doc.page_content for doc in docs])
            # Inject the retrieved context before the user's question
            augmented_user_input = f"Context: {context}\n\nQuestion: {user_input}\n"
        else:
            # No knowledge base available — use the LLM's parametric knowledge only
            augmented_user_input = f"Question: {user_input}\n"

        # Stream the response token-by-token; the "▌" cursor shows the LLM is still typing
        for response in chain.stream({"input": augmented_user_input}):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
