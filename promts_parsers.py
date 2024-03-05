import os
import openai
import datetime
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

class LangChainManager:
    def __init__(self):
        load_dotenv(find_dotenv())
        openai.api_key = os.environ['OPENAI_API_KEY']
        self.current_date = datetime.datetime.now().date()
        self.target_date = datetime.date(2024, 6, 12)
        self.llm_model = self.get_llm_model()

    def get_llm_model(self):
        if self.current_date > self.target_date:
            return "gpt-3.5-turbo"
        else:
            return "gpt-3.5-turbo-0301"

class ChatHandler:
    def __init__(self, temperature=0.0, model="gpt-3.5-turbo"):
        self.chat = ChatOpenAI(temperature=temperature, model=model)

    def generate_response(self, messages):
        return self.chat(messages)

class PromptBuilder:
    @staticmethod
    def build_template(template_string):
        return ChatPromptTemplate.from_template(template_string.strip())

class OutputParser:
    @staticmethod
    def build_output_parser(response_schemas):
        return StructuredOutputParser.from_response_schemas(response_schemas)

if __name__ == "__main__":
    lang_chain_manager = LangChainManager()
    print(f"lang_chain_manager: {lang_chain_manager}\n")
    customer_style = "American English in a calm and respectful tone"
    customer_email = """
    Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! 
    And to make matters worse, the warranty don't cover the cost of cleaning up me kitchen. 
    I need yer help right now, matey!
    """
    template_string = """Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{text}```"""
    prompt_template = PromptBuilder.build_template(template_string)
    customer_messages = prompt_template.format_messages(style=customer_style, text=customer_email)

    chat_handler = ChatHandler(temperature=0.0, model=lang_chain_manager.llm_model)
    #customer_response = chat_handler.generate_response(customer_messages)
    #print(f'response: {customer_response}\n')

    service_reply = """
    Hey there customer, the warranty does not cover cleaning expenses for your kitchen 
    because it's your fault that you misused your blender by forgetting to put the lid on 
    before starting the blender. Tough luck! See ya!
    """
    service_style_pirate = "a polite tone that speaks in English Pirate"
    service_messages = prompt_template.format_messages(style=service_style_pirate, text=service_reply)
    print(f"service messages: {service_messages[0].content}")

    customer_review = """
    This leaf blower is pretty amazing. It has four settings: candle blower, gentle breeze, 
    windy city, and tornado. It arrived in two days, just in time for my wife's anniversary present. 
    I think my wife liked it so much she was speechless. So far I've been the only one using it, 
    and I've been using it every other morning to clear the leaves on our lawn. It's slightly more 
    expensive than the other leaf blowers out there, but I think it's worth it for the extra features.
    """
    
    review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

    response_schemas = [
        ResponseSchema(name="gift", description="Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown."),
        ResponseSchema(name="delivery_days", description="How many days did it take for the product to arrive? If this information is not found, output -1."),
        ResponseSchema(name="price_value", description="Extract any sentences about the value or price, and output them as a comma separated Python list.")
    ]

    output_parser = OutputParser.build_output_parser(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    prompt_template = PromptBuilder.build_template(review_template)
    messages = prompt_template.format_messages(text=customer_review, format_instructions=format_instructions)
    print(f"Messages:{messages[0].content}")

    #response = chat_handler.generate_response(messages)
    #print(response.content)
    #print(type(response.content))

    #output_dict = output_parser.parse(response.content)
    #print(f"type of output_dict: {type(output_dict)}")
    #print(f"delivery days: {output_dict.get('delivery_days')}")