#
import sys
import os
from langchain.document_loaders import OnlinePDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def load_and_split_pdf(url):
    try:
        loader = OnlinePDFLoader(url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        return text_splitter.split_documents(data)
    except Exception as e:
        print(f"Error loading or splitting PDF: {e}")
        return []

def initialize_vectorstore(documents):
    try:
        with SuppressStdout():
            return Chroma.from_documents(documents=documents, embedding=GPT4AllEmbeddings())
    except Exception as e:
        print(f"Error initializing vectorstore: {e}")
        return None

def create_prompt_template():
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    return PromptTemplate(input_variables=["context", "question"], template=template)

def create_retrieval_qa_chain(vectorstore, prompt_template):
    try:
        llm = Ollama(model="llama3:8b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        return RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template},
        )
    except Exception as e:
        print(f"Error creating retrieval QA chain: {e}")
        return None

def main():
    pdf_url = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf"
    all_splits = load_and_split_pdf(pdf_url)
    if not all_splits:
        print("Failed to load and split the PDF. Exiting.")
        return

    vectorstore = initialize_vectorstore(all_splits)
    if not vectorstore:
        print("Failed to initialize vectorstore. Exiting.")
        return

    prompt_template = create_prompt_template()
    qa_chain = create_retrieval_qa_chain(vectorstore, prompt_template)
    if not qa_chain:
        print("Failed to create retrieval QA chain. Exiting.")
        return

    while True:
        query = input("\nQuery: ")
        if query.lower() == "exit":
            break
        if query.strip() == "":
            continue

        try:
            result = qa_chain({"query": query})
            print(result)
        except Exception as e:
            print(f"Error during query processing: {e}")

if __name__ == "__main__":
    main()
