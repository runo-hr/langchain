import os
import openai
import sys
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import SVMRetriever, TFIDFRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


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

    @staticmethod
    def create_vector_store_from_texts(texts, embedding):
        """Create a vector store from texts."""
        smalldb = Chroma.from_texts(texts, embedding=embedding)
        return smalldb


class RetrievalManager:
    @staticmethod
    def similarity_search(vector_store, question, k):
        """Perform similarity search."""
        print(f"\nSimilarity search results for question: '{question}'\n")
        results = vector_store.similarity_search(question, k=k)
        for result in results:
            print(result)
            print()
        return results

    @staticmethod
    def max_marginal_relevance_search(vector_store, question, k, fetch_k):
        """Perform maximum marginal relevance search."""
        print(f"\nMax marginal relevance search results for question: '{question}'\n")
        results = vector_store.max_marginal_relevance_search(question, k=k, fetch_k=fetch_k)
        for result in results:
            print(result)
            print()
        return results


class MetadataRetrievalManager:
    @staticmethod
    def similarity_search_with_metadata(vector_store, question, k, filter_metadata):
        """Perform similarity search with metadata."""
        print(f"\nSimilarity search results with metadata for question: '{question}'\n")
        print(f"The specified metadata is: {filter_metadata}\n")
        results = vector_store.similarity_search(question, k=k, filter=filter_metadata)
        for result in results:
            print(result.metadata)
            print()
        return results


class SelfQueryRetrievalManager:
    @staticmethod
    def create_retriever(llm, vector_store, document_content_description, metadata_field_info, verbose=True):
        """Create a retriever with self-query capabilities."""
        retriever = SelfQueryRetriever.from_llm(
            llm,
            vector_store,
            document_content_description,
            metadata_field_info,
            verbose=verbose
        )
        return retriever

    @staticmethod
    def retrieve_documents(retriever, question):
        """Retrieve documents using self-query retriever."""
        print(f"Retrieving documents for question: '{question}'\n")
        results = retriever.get_relevant_documents(question)
        print()
        for result in results:
            print(result)
            print()
        return results


class ContextualCompressionRetrievalManager:
    @staticmethod
    def create_compression_retriever(base_compressor, base_retriever):
        """Create a contextual compression retriever."""
        return ContextualCompressionRetriever(
            base_compressor=base_compressor,
            base_retriever=base_retriever
        )

    @staticmethod
    def retrieve_documents_with_compression(compression_retriever, question):
        """Retrieve documents using contextual compression retriever."""
        print(f"\nProblem: Compression of retrieved documents\n"
              f"Retrieving documents with compression for question: '{question}'\n"
              "Another approach for improving the quality of retrieved docs is compression.\n"
              "Information most relevant to a query may be buried in a document with a lot of irrelevant text.\n"
              "Passing that full document through your application can lead to more expensive LLM calls and poorer responses.\n"
              "Contextual compression is meant to fix this.\n")
        results = compression_retriever.get_relevant_documents(question)
        for result in results:
            print(result)
            print()
        return results


if __name__ == "__main__":
    persist_directory = 'docs/chroma/'
    embedding = OpenAIEmbeddings()
    vectordb = VectorStoreManager.create_vector_store(embedding, persist_directory)

    texts = [
        """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
        """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
        """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
    ]
    smalldb = VectorStoreManager.create_vector_store_from_texts(texts, embedding)

    question = "Tell me about all-white mushrooms with large fruiting bodies"
    print(f"Problem: Lack of diversity in relevant results.\n"
           "Similarity search doesnt give results about the mushroom being one of the most poisonous.\n"
          "It only brings the most  similar texts in the database.\n"
          )
    RetrievalManager.similarity_search(smalldb, question, k=2)
    RetrievalManager.max_marginal_relevance_search(smalldb, question, k=2, fetch_k=3)
    
    question = "what did they say about regression in the third lecture?"
    MetadataRetrievalManager.similarity_search_with_metadata(vector_store=vectordb, 
                                                             question=question,
                                                             k=3,
                                                             filter_metadata={"source":"docs/cs229_lectures/MachineLearning-Lecture03.pdf"})

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The page from the lecture",
            type="integer",
        ),
    ]
    llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
    retriever = SelfQueryRetrievalManager.create_retriever(
        llm,
        vectordb,
        "Lecture notes",
        metadata_field_info
    )

    question = "what did they say about regression in the third lecture?"
    print(f"Problem: Inferring metadata from the query\n")
    SelfQueryRetrievalManager.retrieve_documents(retriever, question)

    llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetrievalManager.create_compression_retriever(
        base_compressor=compressor,
        base_retriever=vectordb.as_retriever()
    )

    question = "what did they say about matlab?"
    print(f"Problem: Compression of retrieved documents\n")
    ContextualCompressionRetrievalManager.retrieve_documents_with_compression(compression_retriever, question)

    compression_retriever = ContextualCompressionRetrievalManager.create_compression_retriever(
        base_compressor=compressor,
        base_retriever=vectordb.as_retriever(search_type="mmr")
    )
    question = "what did they say about matlab?"
    print(f"Problem: Combining various techniques\nQuestion: '{question}'")
    ContextualCompressionRetrievalManager.retrieve_documents_with_compression(compression_retriever, question)
