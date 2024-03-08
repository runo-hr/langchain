import os
import datetime

import openai

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import WebBaseLoader
from dotenv import load_dotenv, find_dotenv

class LangChainManager:
    def __init__(self):
        load_dotenv(find_dotenv())
        openai.api_key = os.environ['OPENAI_API_KEY']
        self.current_date = datetime.datetime.now().date()
        self.target_date = datetime.date(2024, 6, 12)
        self.llm_model = self.get_llm_model()

    def get_llm_model(self):
        return "gpt-3.5-turbo" if self.current_date > self.target_date else "gpt-3.5-turbo-0301"
    
class DocumentLoaderManager:
    @staticmethod
    def load_pdf(pdf_path):
        loader = PyPDFLoader(pdf_path)
        return loader.load()

    @staticmethod
    def load_youtube(url, save_dir):
        loader = GenericLoader(
            YoutubeAudioLoader([url], save_dir),
            OpenAIWhisperParser()
        )
        return loader.load()

    @staticmethod
    def load_web(url):
        loader = WebBaseLoader(url)
        return loader.load()

if __name__ == "__main__":
    lang_chain_manager = LangChainManager()

    # Loading PDFs
    pdf_path = "data/MachineLearning-Lecture01.pdf"
    pages = DocumentLoaderManager.load_pdf(pdf_path)
    print(len(pages))
    page = pages[0]
    print(page.page_content[0:500])
    print(page.metadata)

    # Loading from YouTube
    url = "https://youtu.be/af-PpBPnp0c?si=sqaiiLqU0nzpi0nO"
    save_dir = "data/youtube/"
    docs = DocumentLoaderManager.load_youtube(url, save_dir)
    print(docs[0].page_content[0:500])

    # Loading from URLs
    url = "https://github.com/basecamp/handbook/blob/master/37signals-is-you.md"
    docs = DocumentLoaderManager.load_web(url)
    print(docs[0].page_content[:500])
