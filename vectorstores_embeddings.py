import os
import datetime
import openai
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
import shutil


class LangChainManager:
    def __init__(self):
        """Initialize the LangChainManager."""
        load_dotenv(find_dotenv())
        openai.api_key = os.environ['OPENAI_API_KEY']
        self.current_date = datetime.datetime.now().date()
        self.target_date = datetime.date(2024, 6, 12)
        self.llm_model = self.get_llm_model()

    def get_llm_model(self):
        """Get the appropriate LLM model based on the current date."""
        return "gpt-3.5-turbo" if self.current_date > self.target_date else "gpt-3.5-turbo-0301"



class DocumentProcessor:
    @staticmethod
    def load_documents(loaders):
        """Load documents from various loaders."""
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        return docs

    @staticmethod
    def split_documents(docs, chunk_size, chunk_overlap):
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(docs)


class VectorStoreManager:
    @staticmethod
    def create_vector_store(documents, embedding, persist_directory):
        """Create a vector store from documents."""
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)  # Remove old database files

        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_directory
        )
        return vectordb

    @staticmethod
    def similarity_search(vector_store, question, k):
        """Perform similarity search on the vector store."""
        return vector_store.similarity_search(question, k=k)



if __name__ == "__main__":
    # Load PDFs
    loaders = [
        PyPDFLoader("data/MachineLearning-Lecture01.pdf"),
        PyPDFLoader("data/MachineLearning-Lecture01.pdf"),
        PyPDFLoader("data/MachineLearning-Lecture02.pdf"),
        PyPDFLoader("data/MachineLearning-Lecture03.pdf")
    ]
    docs = DocumentProcessor.load_documents(loaders)
    print(f"Total number of loaded documents: {len(docs)}\n")

    # Split documents
    chunk_size = 1500
    chunk_overlap = 150
    splits = DocumentProcessor.split_documents(docs, chunk_size, chunk_overlap)
    print(f"Total number of document splits: {len(splits)}\n")

    # Embeddings
    embedding = OpenAIEmbeddings()
    sentence1 = "i like dogs"
    sentence2 = "i like canines"
    sentence3 = "the weather is ugly outside"
    embedding1 = embedding.embed_query(sentence1)
    embedding2 = embedding.embed_query(sentence2)
    embedding3 = embedding.embed_query(sentence3)
    print(f"Similarity between 'dogs' and 'canines': {np.dot(embedding1, embedding2)}")
    print(f"Similarity between 'dogs' and 'weather': {np.dot(embedding1, embedding3)}")
    print(f"Similarity between 'canines' and 'weather': {np.dot(embedding2, embedding3)}\n")

    # Vector stores
    persist_directory = 'data/chroma/'
    vectordb = VectorStoreManager.create_vector_store(splits, embedding, persist_directory)
    print(f"Total number of documents in the vector store: {vectordb._collection.count()}\n")

    # Similarity search
    question = "is there an email I can ask for help"
    k = 3
    similar_docs = VectorStoreManager.similarity_search(vectordb, question, k)
    print(f"question: {question}")
    print(f"Number of similar documents found: {len(similar_docs)}")
    print(f"Page content of the most similar document:\n {similar_docs[0].page_content}\n\n")

    # Failure modes
    question1 = "what did they say about matlab?"
    docs1 = vectordb.similarity_search(question1, k=5)
    print(f"question: {question1}")
    print("\nFailure mode 1: Duplicate Documents")
    print("In this case, the same document has been loaded multiple times, leading to redundant search results.\n")
    print(f"result 1: {docs1[0]}\n")
    print(f"result 2: {docs1[1]}\n\n")

    question2 = "what did they say about regression in the third lecture?"
    docs2 = vectordb.similarity_search(question2, k=5)
    print(f"question: {question2}")
    print("\nFailure mode 2: Cross Lecture Search")
    print("Here, the search query references a specific lecture, but the results include content from other lectures as well.\n")
    for doc in docs2:
        print(doc.metadata)
    print(f"\nPage content of the wrong document: {docs2[4].page_content}\n")
