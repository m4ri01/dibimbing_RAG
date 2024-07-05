import streamlit as st
import os
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

# Initialize session state for document store and pipeline store if they don't exist
if 'document_store' not in st.session_state:
    st.session_state.document_store = InMemoryDocumentStore()

if 'pipeline_store' not in st.session_state:
    pipeline_store = Pipeline()
    pipeline_store.add_component("converter", PyPDFToDocument())
    pipeline_store.add_component("cleaner", DocumentCleaner())
    pipeline_store.add_component("embedder", SentenceTransformersDocumentEmbedder())
    pipeline_store.add_component("writer", DocumentWriter(document_store=st.session_state.document_store,policy=DuplicatePolicy.SKIP))
    pipeline_store.connect("converter", "cleaner")
    pipeline_store.connect("cleaner", "embedder")
    pipeline_store.connect("embedder", "writer")
    st.session_state.pipeline_store = pipeline_store

if 'pipeline_retrieve' not in st.session_state:
    template = """
    given these documents, answer the question below. Documents:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}
    Question: {{query}}
    """
    pipeline_retrieve = Pipeline()
    pipeline_retrieve.add_component("embedder",SentenceTransformersTextEmbedder())
    pipeline_retrieve.add_component("retriever",InMemoryEmbeddingRetriever(document_store=st.session_state.document_store,top_k=5))
    pipeline_retrieve.add_component("builder",PromptBuilder(template=template))
    pipeline_retrieve.add_component("generator",OllamaGenerator(model="llama2",url="http://localhost:11434/api/generate",generation_kwargs={
        "num_predict":-2,
        "temperature":0.9,
    }))
    pipeline_retrieve.connect("embedder","retriever")
    pipeline_retrieve.connect("retriever","builder")
    pipeline_retrieve.connect("builder","generator")
    st.session_state.pipeline_retrieve = pipeline_retrieve

    

# Retrieve from session state
document_store = st.session_state.document_store
pipeline_store = st.session_state.pipeline_store
pipeline_retrieve = st.session_state.pipeline_retrieve
st.title("HR AI Partner")
st.header("CV Screening using RAG system", divider="rainbow")

st.subheader("Please Upload the Candidate's CV")

with st.form("upload_form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Choose pdf file", type="pdf", accept_multiple_files=True)
    submitted = st.form_submit_button("submit")

    if uploaded_file is not None and submitted:
        if not os.path.exists("cv"):
            os.makedirs("cv")
        for file in uploaded_file:
            with open(os.path.join("cv", file.name), "wb") as f:
                f.write(file.getbuffer())
        st.write("File Uploaded Successfully")

st.subheader("Update Document Store")
st.write("This will update the document store with the uploaded CVs")
if st.button("Update Document Store"):
    # st.session_state.document_store = InMemoryDocumentStore()
    files = [os.path.join("cv", f) for f in os.listdir("cv")]
    pipeline_store.run({"converter": {"sources": files}})
    # st.write(document_store.filter_documents())
    st.write("Document Store Updated Successfully")

st.subheader("Show Document Store")
st.write("This will show all documents in the document store")
if st.button("Show Document Store"):
    st.write(document_store.filter_documents())

st.subheader("RAG from Document Store")
st.write("This will retrieve documents from the document store and use it to augment the answer generation")
query = st.text_input("Enter your question here")
if st.button("Generate Answer"):
    result = pipeline_retrieve.run({
        "embedder":{"text":query},
        "builder":{"query":query}
    })
    st.write(result["generator"]["replies"][0])
    st.write(result)