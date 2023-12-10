
import os
import pinecone
import streamlit as st
from llama_index import ServiceContext
from llama_index.embeddings import LangchainEmbedding
from llama_index.vector_stores import PineconeVectorStore
from llama_index.query_engine import CitationQueryEngine
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext

st.title("QnA on PDF's")

if "messages" not in st.session_state:
    st.session_state.messages = []


open_ai_key = st.sidebar.text_input("OpenAI Key", type="password")
pinecone_key = st.sidebar.text_input("Pinecone Key", type="password")
os.environ["OPENAI_API_KEY"] = open_ai_key


@st.cache_resource
def create_index(data_path):
    """
    Create and return a VectorStoreIndex for PDF documents.

    Parameters:
    - data_path (str): The path to the directory containing PDF documents.

    Returns:
    - VectorStoreIndex: The created index for the PDF documents.
    """
    pinecone.init(api_key=pinecone_key, environment="us-east1-gcp")
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2")
    )
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    pinecone_index = pinecone.Index("pdf-rag")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store)
    documents = SimpleDirectoryReader(data_path).load_data()
    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context, storage_context=storage_context)
    return index


@st.cache_resource
def initializing():
    """
    Initialize Pinecone and create a CitationQueryEngine.

    Returns:
    - CitationQueryEngine: The initialized query engine.
    """
    print("Setting up RAG")
    pinecone.init(api_key=pinecone_key, environment="us-east1-gcp")
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2")
    )
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    pinecone_index = pinecone.Index("pdf-rag")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # create Index, this step is only needed once,
    # index = create_index("data_path/")

    print("LOADING INDEX")
    # Load index from vector store, use this after creating index
    index = VectorStoreIndex.from_vector_store(
        vector_store, service_context)

    print("Initializing Query Engine")
    query_engine = CitationQueryEngine.from_args(
        index,
        similarity_top_k=2,
        # here we can control how granular citation sources are, the default is 512
        citation_chunk_size=512,
    )
    return query_engine


if open_ai_key and pinecone_key:
    query_engine = initializing()
elif not open_ai_key and not pinecone_key:
    st.error("Please input OpenAI key and Pinecone Key")
elif not pinecone_key:
    st.error("Please input Pinecone Key")
elif not open_ai_key:
    st.error("Please input OpenAI key")


if open_ai_key and pinecone_key:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        print("Generating Response")
        response = query_engine.query(prompt)

        with st.chat_message("assistant"):
            st.markdown(f"{response}")

        st.markdown(f":red[**CITATIONS:**]\n")
        for source in response.source_nodes:
            file_name = source.metadata["file_name"]
            page_number = source.metadata["page_label"]
            reference_text = source.node.get_text()
            with st.chat_message("assistant"):
                st.markdown(
                    f"The following citation belongs to the file :green[**{file_name}**] and page number :green[**{page_number}**]")
                st.markdown(source.node.get_text())

        st.session_state.messages.append(
            {"role": "assistant", "content": response})
