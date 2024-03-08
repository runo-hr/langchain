import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter, MarkdownHeaderTextSplitter
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv

class LangChainManager:
    def __init__(self):
        load_dotenv(find_dotenv())
        openai.api_key = os.environ['OPENAI_API_KEY']

class DocumentSplitterManager:
    @staticmethod
    def split_recursive(text, chunk_size, chunk_overlap, separators=None):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
        return splitter.split_text(text)

    @staticmethod
    def split_character(text, chunk_size, chunk_overlap, separator=' '):
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator)
        return splitter.split_text(text)

    @staticmethod
    def split_token(text, chunk_size, chunk_overlap):
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)

    @staticmethod
    def split_markdown(text, headers_to_split_on):
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        return splitter.split_text(text)

if __name__ == "__main__":
    lang_chain_manager = LangChainManager()

    # Recursive splitting
    text1 = 'abcdefghijklmnopqrstuvwxyz'
    print(f"Recursive splitting result: {DocumentSplitterManager.split_recursive(text1, chunk_size=26, chunk_overlap=4)}\n")

    # Character splitting
    text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
    print(f"Character splitting result: {DocumentSplitterManager.split_character(text2, chunk_size=26, chunk_overlap=4)}\n")

    # Token splitting
    text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    print(f"Token splitting result: {DocumentSplitterManager.split_token(text3, chunk_size=26, chunk_overlap=4)}\n")

    # Markdown header splitting
    markdown_document = """# Title\n\n ## Chapter 1\n\n Hi this is Jim\n\n Hi this is Joe\n\n ### Section \n\n Hi this is Lance \n\n ## Chapter 2\n\n Hi this is Molly"""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    print(f"Markdown header splitting result: {DocumentSplitterManager.split_markdown(markdown_document, headers_to_split_on)}\n")

    # Loading from PDF
    loader = PyPDFLoader("data/MachineLearning-Lecture01.pdf")
    pages = loader.load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=150, length_function=len)
    docs = text_splitter.split_documents(pages)
    print(f"Number of documents after splitting: {len(docs)}")
    print(f"Number of pages in the original document: {len(pages)}\n")
