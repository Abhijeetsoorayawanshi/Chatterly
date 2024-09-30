import os
import warnings
from PyPDF2 import PdfReader
import streamlit as st 
import pickle
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize HuggingFaceEmbeddings with the SentenceTransformer model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Initialize Hugging Face Question Answering pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Custom CSS for removing background color and positioning the input and submit button
def custom_css():
    st.markdown("""
        <style>
        /* Remove background colors from conversation history */
        .user-input{
            text-align : right;
            background-color: transparent !important;
            padding: 10px;
            margin: 5px 0;
            border: none;
        }
        .bot-response{
            text-align : left;
            background-color: transparent !important;
            padding: 10px;
            margin: 5px 0;
            border: none;
        }

        /* Align the submit button to the right of the input field */
        .input-container {
            display: flex;
            justify-content: flex-start;
            width: 100%;
        }
        .input-field {
            flex-grow: 1;
        }
        .submit-button {
            margin-left: 10px;
        }

        /* Ensure conversation history fits within the content area */
        .conversation-container {
            max-height: 70vh;
            overflow-y: auto;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title('Welcome to Chatterly')
    st.markdown(''' 
    ## About 
    This app is an LLM-powered chatbot built using 
    - Streamlit           
    - Langchain
    - Hugging Face Transformers
    ''')
    add_vertical_space(5)
    st.write('By Abhijeet Kumar')

def main():
    # load_dotenv()
    st.header("Ask from your PDF")
    
    # Inject custom CSS
    custom_css()

    # Upload your PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        # Convert the PDF to text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        ) 
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorstore = pickle.load(f)
        else:
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)
        st.write("**Hi... I am chatterly. I have read the document you provided. I will be happy to solve your queries from this document.**")

        # Initialize session state to keep track of conversation
        if "conversation" not in st.session_state:
            st.session_state.conversation = []

        # Conversation history container
        st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
        
        # Loop through conversation history
        # for idx, (user_input, response) in enumerate(st.session_state.conversation):
        #     st.markdown(f'<div class="user-input">{user_input}</div>', unsafe_allow_html=True)
        #     st.markdown(f'<div class="bot-response">Chatterly: {response}</div>', unsafe_allow_html=True)
        for idx, (user_input, response) in enumerate(st.session_state.conversation):
            st.markdown(f'<div class="user-input">{user_input}</div>', unsafe_allow_html=True)
            
            # Add image tag for Chatterly icon
            chatterly_icon = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTKKS9YDJFcRRxNkw0vn-p_Lu6wbntU2S0Pjw&s'  
            st.markdown(f'''
                <div class="bot-response">
                    <img src="{chatterly_icon}" alt="Chatterly" style="width: 25px; vertical-align: middle;margin-right: 10px;border-radius: 50%;"/>
                      {response}
                </div>
            ''', unsafe_allow_html=True)
        
        # st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Input field and submit button aligned horizontally
        st.markdown('<div class="input-container">', unsafe_allow_html=True)

        # Input text field
        latest_question = st.text_input("Ask me anything about the document:", key=f"input_{len(st.session_state.conversation)}", placeholder="Type your question here...", label_visibility="collapsed", help="Type your query here.")
        
        # Submit button next to input field
        submit_button = st.button("Submit", key=f"submit_{len(st.session_state.conversation)}", use_container_width=False)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Handle the latest question when the user submits it
        if submit_button and latest_question:
            with st.spinner('Processing...'):
                docs = vectorstore.similarity_search(query=latest_question)
                context = " ".join([doc.page_content for doc in docs]) if docs else ""

                if context:
                    result = qa_pipeline(question=latest_question, context=context)
                    answer = result['answer']
                else:
                    answer = "No relevant documents found."
                
                # Append the question and answer to the conversation history
                st.session_state.conversation.append((latest_question, answer))


if __name__ == "__main__":
    main()







# import os
# import warnings
# from PyPDF2 import PdfReader
# import streamlit as st 
# import pickle
# from dotenv import load_dotenv
# from streamlit_extras.add_vertical_space import add_vertical_space
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from transformers import pipeline

# # Suppress FutureWarnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Suppress symlink warning
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# # Initialize HuggingFaceEmbeddings with the SentenceTransformer model
# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# # Initialize Hugging Face Question Answering pipeline
# qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# with st.sidebar:
#     st.title('Welcome to Chatterly')
#     st.markdown(''' 
#     ## About 
#     This app is an LLM-powered chatbot built using 
#     - Streamlit           
#     - Langchain
#     - Hugging Face Transformers
#     ''')
#     add_vertical_space(5)
#     st.write('By Abhijeet Kumar')

# def main():
#     load_dotenv()
#     st.header("Ask from your PDF")
    
#     # Upload your PDF file
#     pdf = st.file_uploader("Upload your PDF", type='pdf')
    
#     if pdf is not None:
#         pdf_reader = PdfReader(pdf)
        
#         # Convert the PDF to text
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
        
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         ) 
#         chunks = text_splitter.split_text(text=text)

#         store_name = pdf.name[:-4]
#         if os.path.exists(f"{store_name}.pkl"):
#             with open(f"{store_name}.pkl", "rb") as f:
#                 vectorstore = pickle.load(f)
#         else:
#             vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
#             with open(f"{store_name}.pkl", "wb") as f:
#                 pickle.dump(vectorstore, f)
        
#         st.write("Hi... I am chatterly. I have read the document you provided. I will be happy to solve your queries from this document.")

#         # Initialize session state to keep track of conversation
#         if "conversation" not in st.session_state:
#             st.session_state.conversation = []

#         # Loop through conversation history
#         for idx, (user_input, response) in enumerate(st.session_state.conversation):
#             st.write(f"**You :** {user_input}")
#             st.write(f"**Chatterly:** {response}")

#         # Display new input field for each question asked
#         latest_question = st.text_input("Question", key=f"input_{len(st.session_state.conversation)}")
#         submit_button = st.button("Submit", key=f"submit_{len(st.session_state.conversation)}")

#         # Handle the latest question when the user submits it
#         if submit_button and latest_question:
#             docs = vectorstore.similarity_search(query=latest_question)
#             context = " ".join([doc.page_content for doc in docs]) if docs else ""
            
#             if context:
#                 result = qa_pipeline(question=latest_question, context=context)
#                 answer = result['answer']
#             else:
#                 answer = "No relevant documents found."
            
#             # Append the question and answer to the conversation history
#             st.session_state.conversation.append((latest_question, answer))

# if __name__ == "__main__":
#     main()



