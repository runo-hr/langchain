import os
import datetime

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

class LangChainManager:
    @staticmethod
    def initialize_llm():
        """Initialize the LLM."""
        current_date = datetime.datetime.now().date()
        if current_date < datetime.date(2023, 9, 2):
            llm_name = "gpt-3.5-turbo-0301"
        else:
            llm_name = "gpt-3.5-turbo"
        print(llm_name)
        return ChatOpenAI(model_name=llm_name, temperature=0)

    @staticmethod
    def note_langchain_plus():
        """Note about LangChain Plus platform."""
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
    def create_vector_store(embedding, persist_directory):
        """Create a vector store."""
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        print(f"Total number of documents in the vector store: {vectordb._collection.count()}\n")
        return vectordb


class PromptManager:
    @staticmethod
    def create_qa_prompt(template):
        """Create a prompt for QA chain."""
        return PromptTemplate.from_template(template)


class QuestionAnsweringManager:
    @staticmethod
    def build_qa_chain(llm, vectordb, chain_type="stuff", prompt=None):
        """Build a RetrievalQA chain."""
        kwargs = {}
        if prompt:
            kwargs["prompt"] = prompt
        return RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type=chain_type,
            chain_type_kwargs=kwargs
        )

    @staticmethod
    def run_retrieval_qa_chain(llm, vectordb, question):
        """Run RetrievalQA chain."""
        qa_chain = QuestionAnsweringManager.build_qa_chain(llm, vectordb)
        result = qa_chain({"query": question})
        print(f"\nRetrievalQA chain result for question: '{question}': \n{result['result']}\n")

    @staticmethod
    def run_prompted_qa_chain(llm, vectordb, prompt, question):
        """Run RetrievalQA chain with a prompt."""
        qa_chain = QuestionAnsweringManager.build_qa_chain(llm, vectordb, prompt=prompt)
        result = qa_chain({"query": question})
        print(f"\nPrompted RetrievalQA chain result for question: '{question}': \n{result['result']}\n")
        print(f"Source document: {result['source_documents'][0]}\n")

    @staticmethod
    def run_retrieval_qa_specify_chain_type(llm, vectordb, chain_type, question):
        """Run RetrievalQA chain with specified chain type."""
        qa_chain = QuestionAnsweringManager.build_qa_chain(llm, vectordb, chain_type=chain_type)
        result = qa_chain({"query": question})
        print(f"\n RetrievalQA chain with chain type '{chain_type}' result for question: '{question}': \n{result['result']}\n")

    @staticmethod
    def demonstrate_qa_limitations(llm, vectordb):
        """Demonstrate limitations of RetrievalQA."""
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever()
        )

        print("\n### RetrievalQA limitations\n")
        question1 = "Is probability a class topic?"
        result1 = qa_chain({"query": question1})
        print(f"Result for question: '{question1}': \n{result1['result']}")

        question2 = "why are those prerequisites needed?"
        result2 = qa_chain({"query": question2})
        print(f"Result for question: '{question2}': \n{result2['result']}\n")

        print("""NOTE: The LLM response varies. Some responses **do** include a reference to probability which might be gleaned from referenced documents. 
        The point is simply that the model does not have access to past questions or answers, 
        this will be covered in the next section.
        """)


if __name__ == "__main__":
    persist_directory = 'docs/chroma/'
    embedding = OpenAIEmbeddings()
    vectordb = VectorStoreManager.create_vector_store(embedding, persist_directory)

    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm = LangChainManager.initialize_llm()

    question = "What are major topics for this class?"
    QuestionAnsweringManager.run_retrieval_qa_chain(llm, vectordb, question)

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    # context = "Example context here."
    prompt = PromptManager.create_qa_prompt(template)
    QuestionAnsweringManager.run_prompted_qa_chain(llm, vectordb, prompt, question)

    question = "Is probability a class topic?"
    QuestionAnsweringManager.run_retrieval_qa_specify_chain_type(llm, vectordb, chain_type="map_reduce", question=question)

    QuestionAnsweringManager.demonstrate_qa_limitations(llm, vectordb)
    LangChainManager.note_langchain_plus()
