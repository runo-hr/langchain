import os
import openai
import datetime
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import RouterOutputParser, LLMRouterChain
from langchain.chains import (
    LLMChain,
    SimpleSequentialChain,
    SequentialChain   
)

_ = load_dotenv(find_dotenv())  # Read local .env file

class LangChainManager:
    def __init__(self):
        load_dotenv(find_dotenv())
        openai.api_key = os.environ['OPENAI_API_KEY']
        self.current_date = datetime.datetime.now().date()
        self.target_date = datetime.date(2024, 6, 12)
        self.llm_model = self.get_llm_model()

    def get_llm_model(self):
        return "gpt-3.5-turbo" if self.current_date > self.target_date else "gpt-3.5-turbo-0301"

class ChainsDemo:
    df = pd.read_csv('data/Data.csv')
    @staticmethod
    def demonstrate_llm_chain(llm_model):
        llm = ChatOpenAI(temperature=0.9, model=llm_model)
        prompt = ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")
        chain = LLMChain(llm=llm, prompt=prompt)
        product = "Queen Size Sheet Set"
        print(chain.run(product))

    @staticmethod
    def demonstrate_simple_sequential_chain(llm_model):
        llm = ChatOpenAI(temperature=0.9, model=llm_model)
        first_prompt = ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")
        chain_one = LLMChain(llm=llm, prompt=first_prompt)
        second_prompt = ChatPromptTemplate.from_template("Write a 20 words description for the following company:{company_name}")
        chain_two = LLMChain(llm=llm, prompt=second_prompt)
        overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)
        print(overall_simple_chain.run("Queen Size Sheet Set"))

    @staticmethod
    def demonstrate_sequential_chain(llm_model):
        llm = ChatOpenAI(temperature=0.9, model=llm_model)
        first_prompt = ChatPromptTemplate.from_template("Translate the following review to english:\n\n{Review}")
        chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")
        second_prompt = ChatPromptTemplate.from_template("Can you summarize the following review in 1 sentence:\n\n{English_Review}")
        chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")
        third_prompt = ChatPromptTemplate.from_template("What language is the following review:\n\n{Review}")
        chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")
        fourth_prompt = ChatPromptTemplate.from_template("Write a follow up response to the following summary in the specified language:\n\nSummary: {summary}\n\nLanguage: {language}")
        chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")
        fifth_prompt = ChatPromptTemplate.from_template("translate the follow up response to english: \n\follow up response: {followup_message}")
        chain_five = LLMChain(llm=llm, prompt=fifth_prompt, output_key="translated_followup_message")
        overall_chain = SequentialChain(
            chains=[chain_one, chain_two, chain_three, chain_four, chain_five],
            input_variables=["Review"],
            output_variables=["English_Review", "language", "summary", "followup_message", "translated_followup_message"],
            verbose=True
        )
        review = ChainsDemo.df.Review[5]
        print(overall_chain(review))

    @staticmethod
    def demonstrate_router_chain(llm_model):
        llm = ChatOpenAI(temperature=0, model=llm_model)
        physics_template = "You are a very smart physics professor. You are great at answering questions about physics in a concise and easy to understand manner. When you don't know the answer to a question you admit that you don't know. Here is a question:\n{input}"
        math_template = "You are a very good mathematician. You are great at answering math questions. You are so good because you are able to break down hard problems into their component parts, answer the component parts, and then put them together to answer the broader question. Here is a question:\n{input}"
        history_template = "You are a very good historian. You have an excellent knowledge of and understanding of people, events and contexts from a range of historical periods. You have the ability to think, reflect, debate, discuss and evaluate the past. You have a respect for historical evidence and the ability to make use of it to support your explanations and judgements. Here is a question:\n{input}"
        computerscience_template = "You are a successful computer scientist. You have a passion for creativity, collaboration, forward-thinking, confidence, strong problem-solving capabilities, understanding of theories and algorithms, and excellent communication skills. You are great at answering coding questions. You are so good because you know how to solve a problem by describing the solution in imperative steps that a machine can easily interpret and you know how to choose a solution that has a good balance between time complexity and space complexity. Here is a question:\n{input}"
        prompt_infos = [
            {"name": "physics", "description": "Good for answering questions about physics", "prompt_template": physics_template},
            {"name": "math", "description": "Good for answering math questions", "prompt_template": math_template},
            {"name": "History", "description": "Good for answering history questions", "prompt_template": history_template},
            {"name": "computer science", "description": "Good for answering computer science questions", "prompt_template": computerscience_template}
        ]
        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = ChatPromptTemplate.from_template(template=prompt_template)
            chain = LLMChain(llm=llm, prompt=prompt)
            destination_chains[name] = chain  
        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        default_prompt = ChatPromptTemplate.from_template("{input}")
        default_chain = LLMChain(llm=llm, prompt=default_prompt)
        MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a language model select the model prompt best suited for the input. You will be given the names of the available prompts and a description of what the prompt is best suited for. You may also revise the original input if you think that revising it will ultimately lead to a better response from the language model.\n\n<< FORMATTING >>\nReturn a markdown code snippet with a JSON object formatted to look like:\n```json\n{{{{\n    "destination": string \ name of the prompt to use or "DEFAULT"\n    "next_inputs": string \ a potentially modified version of the original input\n}}}}\n```\n\nREMEMBER: "destination" MUST be one of the candidate prompt names specified below OR it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.\nREMEMBER: "next_inputs" can just be the original input if you don't think any modifications are needed.\n\n<< CANDIDATE PROMPTS >>\n{destinations}\n\n<< INPUT >>\n{{input}}\n\n<< OUTPUT (remember to include the ```json)>>"""
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(template=router_template, input_variables=["input"], output_parser=RouterOutputParser())
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)
        chain = MultiPromptChain(router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain, verbose=True)
        print(chain.run("What is black body radiation?"))
        print(chain.run("what is 2 + 2"))
        print(chain.run("what is dna"))

if __name__ == "__main__":
    lang_chain_manager = LangChainManager()
    
    ChainsDemo.demonstrate_llm_chain(lang_chain_manager.llm_model)
    ChainsDemo.demonstrate_simple_sequential_chain(lang_chain_manager.llm_model)
    ChainsDemo.demonstrate_sequential_chain(lang_chain_manager.llm_model)
    ChainsDemo.demonstrate_router_chain(lang_chain_manager.llm_model)
