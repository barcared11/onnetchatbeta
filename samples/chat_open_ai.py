import os
import pandas as pd
import openai
import matplotlib.pyplot as plt
import streamlit as st
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader

import pdfminer
import textract

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def main():
    st.sidebar.title("Transformers Chatbot")
    st.sidebar.write("Type your question in the input box below.")

    # Step 2: Save to .txt and reopen (helps prevent issues)
    file = "/Users/amarahmed/Downloads/Interview.txt"
    file_pdf = "/Users/amarahmed/Desktop/reelTour_/sodapdf-converted_interview.pdf"

    loader = PyPDFLoader(file_pdf)
    document = loader.load_and_split()

    doc = textract.process(file_pdf)

    # Step 2: Save to .txt and reopen (helps prevent issues)
    with open(file, 'w') as f:
        f.write(doc.decode('utf-8'))

    with open(file, 'r') as f:
        text = f.read()

    # Step 3: Create function to count tokens
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=512,
        chunk_overlap=24,
        length_function=count_tokens,
    )

    chunks = text_splitter.create_documents([text])
    token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

    # Create a DataFrame from the token counts
    df = pd.DataFrame({'Token Count': token_counts})

    # Create a histogram of the token count distribution
    fig, ax = plt.subplots()
    df.hist(bins=40, ax=ax)

    # Show the plot
    st.pyplot(fig)

    # Get embedding model
    embeddings = OpenAIEmbeddings()

    # Create vector database
    db = FAISS.from_documents(chunks, embeddings)
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

    query = st.text_input("Ask a question")

    if query.strip().lower() == 'exit':
        st.write("Thank you for using the Transformers chatbot!")
        return

    docs = db.similarity_search(query)

    chain.run(input_documents=docs, question=query)
    openai = OpenAI(temperature=0.1)
    qa = ConversationalRetrievalChain.from_llm(openai, db.as_retriever())

    chat_history = []

    if query:
        result = qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, result['answer']))