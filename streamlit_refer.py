import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback

import tempfile

def main():
    st.set_page_config(
        page_title="DirChat",
        page_icon=":robot_face:"
    )

    st.title("_나만의 Data 분석:red[ Chat]_ :robot_face:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        if uploaded_files:
            files_text = get_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            vectorstore = get_vectorstore(text_chunks)

            st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        else:
            st.session_state.conversation = get_default_conversation_chain(openai_api_key)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "왼쪽 화면에 파일 Upload or OpenAI API key를 입력해주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                with get_openai_callback() as cb:
                    if isinstance(chain, ConversationalRetrievalChain):
                        result = chain({"question": query})
                        response = result['answer']
                        source_documents = result['source_documents']
                    else:
                        response = chain.predict(input=query)
                        source_documents = []

                st.markdown(response)
                if source_documents:
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents[:3]:  # 최대 3개의 문서만 표시
                            st.markdown(doc.metadata['source'], help=doc.page_content)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []

    for doc in docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=doc.name.split('.')[-1]) as temp_file:
            temp_file.write(doc.getvalue())
            logger.info(f"Uploaded {doc.name}")

            if '.pdf' in doc.name:
                loader = PyPDFLoader(temp_file.name)
            elif '.docx' in doc.name:
                loader = Docx2txtLoader(temp_file.name)
            elif '.pptx' in doc.name:
                loader = UnstructuredPowerPointLoader(temp_file.name)
            else:
                continue

            documents = loader.load_and_split()
            doc_list.extend(documents)

    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

def get_default_conversation_chain(openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(memory_key='history', return_messages=False),
        verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()
