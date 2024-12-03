# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub

# # Set up authorization headers (you will need the keys in your secrets.toml)
# header = {
#     "authorization": st.secrets["HUGGINGFACEHUB_API_TOKEN"],  # API key from Hugging Face
#     "content-type": "application/json"
# }

# # Function to extract text from PDF documents
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Function to split extracted text into chunks for vectorization
# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# # Function to get the vector store using embeddings
# def get_vectorstore(text_chunks):
#     # Uncomment the embeddings you want to use

#     # OpenAI Embeddings (requires API key)
#     # embeddings = OpenAIEmbeddings()

#     # Hugging Face Instructor embeddings (requires Hugging Face API Key)
#     # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")  # Free model

#     # Hugging Face Sentence Transformers (free models, no API key required)
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Free model

#     # LLaMA or other Hugging Face-based models (requires API key for Hugging Face)
#     # embeddings = HuggingFaceInstructEmbeddings(model_name="meta-llama/LLaMA-7B")  # Requires setup

#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

# # Function to create the conversation chain
# def get_conversation_chain(vectorstore):
#     # Uncomment the LLMs you want to use

#     # OpenAI Chat (requires OpenAI API key)
#     # llm = ChatOpenAI()

#     # Hugging Face FLAN-T5 (free model, no API key required)
#     # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})

#     # Open Source Hugging Face-based LLM (free model, no API key required)
#     llm = HuggingFaceHub(repo_id="bigscience/bloom-560m", model_kwargs={"temperature": 0.7, "max_length": 512})

#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

# # Handle the user input and process responses
# def handle_userinput(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# # Main function to initialize the streamlit app
# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with multiple PDFs :books:")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # Get pdf text
#                 raw_text = get_pdf_text(pdf_docs)

#                 # Get the text chunks
#                 text_chunks = get_text_chunks(raw_text)

#                 # Create vector store
#                 vectorstore = get_vectorstore(text_chunks)

#                 # Create conversation chain
#                 st.session_state.conversation = get_conversation_chain(vectorstore)


# if __name__ == '__main__':
#     main()

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def validate_input(func):
    """
    Decorator to validate input before the main function logic.
    """
    def wrapper(text, *args, **kwargs):
        if not isinstance(text, str) or len(text) == 0:
            raise ValueError("Input must be a non-empty string")
        return func(text, *args, **kwargs)
    return wrapper


@validate_input
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # Initialize the sentence transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Generate embeddings for the text chunks
    embeddings = model.encode(text_chunks, convert_to_tensor=False)
    
    # Convert embeddings to numpy array for FAISS compatibility
    embeddings = np.array(embeddings)
    
    # Initialize the FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    
    # Add embeddings to the FAISS index
    index.add(embeddings)
    
    return index


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Memory to hold the conversation
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    # Initialize conversation chain (retriever is optional, so skipped here)
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'input': user_question})
    st.session_state.chat_history = response.get('memory', [])
    
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.header("Chat with PDFs :books:")
    
    user_question = st.text_input("Ask a question about your document:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get text from PDFs
                raw_text = get_pdf_text(pdf_docs)
                
                # Get text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Create FAISS vectorstore
                vectorstore = get_vectorstore(text_chunks)
                
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Processing complete!")


if __name__ == '__main__':
    main()
