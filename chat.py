import os
import datetime
import openai

from langchain.vectorstores import Chroma, DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys
sys.path.append('../..')

import param
import panel as pn
pn.extension()

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


class LangChainManager:
    @staticmethod
    def initialize_llm():
        """Initialize the Language Model."""
        current_date = datetime.datetime.now().date()
        llm_name = "gpt-3.5-turbo-0301" if current_date < datetime.date(2023, 9, 2) else "gpt-3.5-turbo"
        return ChatOpenAI(model_name=llm_name, temperature=0)

    @staticmethod
    def note_langchain_plus():
        """Provide information about LangChain Plus platform."""
        print("""If you wish to experiment on the `LangChain plus platform`:
        
        * Go to [langchain plus platform](https://www.langchain.plus/) and sign up
        * Create an API key from your account's settings
        * Use this API key in the code below
        * uncomment the code  
        Note, the endpoint in the video differs from the one below. Use the one below.

        #import os
        #os.environ["LANGCHAIN_TRACING_V2"] = "true"
        #os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
        #os.environ["LANGCHAIN_API_KEY"] = "..." # replace dots with your api key
        """)


class VectorStoreManager:
    @staticmethod
    def create_chroma_vector_store(embedding, persist_directory):
        """Create a vector store."""
        return Chroma(persist_directory=persist_directory, embedding_function=embedding)

    @staticmethod
    def create_docarray_vector_store(documents, embedding):
        vectordb = DocArrayInMemorySearch.from_documents(documents=documents, embedding=embedding)
        return vectordb
    
    @staticmethod
    def create_retriever(vectordb, k=4, search_type="similarity"):
        retriever = vectordb.as_retriever(search_type=search_type, search_kwargs={"k": k})
        return retriever


class PromptManager:
    @staticmethod
    def create_qa_prompt(template):
        """Create a prompt for QA chain."""
        return PromptTemplate.from_template(template)


class MemoryManager:
    @staticmethod
    def create_conversation_memory(memory_key, return_messages=True):
        """Create a conversation buffer memory."""
        return ConversationBufferMemory(memory_key=memory_key, return_messages=return_messages)


class ChainManager:
    @staticmethod
    def build_qa_chain(llm, vectordb, prompt=None):
        """Build a RetrievalQA chain."""
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template) if prompt is None else prompt
        
        return RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    @staticmethod
    def build_conversational_chain(llm, retriever, memory, chain_type="stuff", return_source_documents=False, return_generated_question=False):
        """Build a ConversationalRetrievalChain."""
        
        return ConversationalRetrievalChain.from_llm(llm=llm, 
                                                     chain_type=chain_type,
                                                     retriever=retriever, 
                                                     memory=memory,
                                                     return_source_documents=return_source_documents,
                                                     return_generated_question=return_generated_question)


class DataLoader:
    @staticmethod
    def load_documents(file, chunk_size=1000, chunk_overlap=150):
        """Load and split documents."""
        loader = PyPDFLoader(file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

class DocumentsChatbot:
    def __init__(self):
        self.loaded_file = "docs/cs229_lectures/MachineLearning-Lecture01.pdf"
    
    def load_bot(self, chain_type="stuff", k=4, search_type="similarity"):
        documents = DataLoader.load_documents(file=self.loaded_file)
        embeddings = OpenAIEmbeddings()
        
        vectordb = VectorStoreManager.create_docarray_vector_store(documents=documents, embedding=embeddings)
        retriever = VectorStoreManager.create_retriever(vectordb=vectordb, search_type=search_type, k=k)
        
        memory = MemoryManager.create_conversation_memory(memory_key="chat_history")

        llm = LangChainManager.initialize_llm()

        bot = ChainManager.build_conversational_chain(llm=llm, 
                                                      chain_type=chain_type,
                                                      retriever=retriever, 
                                                      memory=memory
                                                      )
        return bot

if __name__ == "__main__":
    persist_directory = 'docs/chroma/'
    embedding = OpenAIEmbeddings()
    vectordb = VectorStoreManager.create_chroma_vector_store(embedding=embedding, persist_directory=persist_directory)

    llm = LangChainManager.initialize_llm()

    qa_chain= ChainManager.build_qa_chain(llm=llm, vectordb=vectordb)

    question = "Is probability a class topic?"
    followup_question = "why are those prerequesites needed?"

    result = qa_chain({"query": question})
    followup_result = qa_chain({"query": followup_question})
    print(f"\n qa_chain result keys: {result.keys()}\n")
    print(f"\n qa_chain (without memory) result for '{question}': {result['result']}\n")
    print(f"\n qa_chain (without memory) result for '{followup_question}': {followup_result['result']}\n")

    memory = MemoryManager.create_conversation_memory(memory_key="chat_history")
    retriever = vectordb.as_retriever()
    conv_chain = ChainManager.build_conversational_chain(llm=llm, retriever=retriever, memory=memory)
    
    result = conv_chain({"question": question})
    followup_result = conv_chain({"question": followup_question})
    print(f"\n conv_chain result keys: {result.keys()}\n")
    print(f"\n conv_chain (with memory) result for '{question}': {result['answer']}\n")
    print(f"\n conv_chain (with memory) result for '{followup_question}': {followup_result['answer']}\n")

    bot = DocumentsChatbot()
    chatbot = bot.load_bot()
    result = chatbot({"question": question})
    followup_result = chatbot({"question": followup_question})
    print(f"\n chatbot result keys: {result.keys()}\n")
    print(f"\n chatbot (with memory) result for '{question}': {result['answer']}\n")
    print(f"\n chatbot (with memory) result for '{followup_question}': {followup_result['answer']}\n")
    print(f"\n chat history: {result.get('chat_history')}")