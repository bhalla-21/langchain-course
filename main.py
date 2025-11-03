import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    # print(os.environ.get("GOOGLE_API_KEY"))
    information = """Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2021; as of October 2025, Forbes estimates his net worth to be around $500 billion.

    Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he had obtained Canadian citizenship at birth through his Canadian-born mother. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen."""

    summary_template = """Given the info {information} about a person, I want you to create :
    1. A short summary
    2. Two very interesting things about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # This is how we initialize object of prompt template class, we give it the list of keys (information variable or any other variable), so that they can be plugged at runtime
    # Formats input text into a prompt string
    # THis enhances visibility - since prompts are first class citizens in LC, and they are logged properly, easier debugging. Safer against prompt injections. Different from fstrings, they just plug inputs

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    # this is how we interact with LLMs , checked class definition , copied from there. LC is going to search the envv variable
    # Check source code of ChatGoogleGenerativeAI and see how it happens there
    # THis is going to be called under the hood to make API calls to gOogle AI sdk, with our key

    # If I want to run locally downloaded Ollama Gemma3 model- uncomment the code
    # llm = ChatOllama(temperature = 0, model = 'gemma3:270m')

    # Now we create our first chain
    chain = (
        summary_prompt_template | llm
    )  # Using LCEL - we create a chain by composing a prompt template & LLM. Input from prompt template propogates into LLM. Pipe operator creates a new runnable chain - Left -> Right component. We invoke this runnable object
    response = chain.invoke(
        input={"information": information}
    )  # Runnable method has this invoke method - we give it input (info key)
    print(response.content)


if __name__ == "__main__":
    main()




#iF WE debug the response with breakpoint in the main function, we see - response is of type ai message AIMessage(content="Here's the information about Elon Musk:\n\n### 1. Short Summary\n\nElon Musk is a South African-born entrepreneur and businessman, currently the wealthiest person in the world with an estimated net worth of $500 billion as of October 2025. He is renowned for leading major companies like Tesla, SpaceX, Twitter, and xAI. After co-founding early successful ventures such as Zip2 and PayPal, he became an American citizen in 2002, having previously held Canadian citizenship by birth.\n\n### 2. Two Very Interesting Things\n\n1.  **Complex Citizenship Journey:** Despite being born in South Africa, he obtained Canadian citizenship at birth through his Canadian-born mother and emigrated to Canada in 1989, before eventually becoming an American citizen in 2002. This means he's held citizenship from at least two, potentially three, different countries over his lifetime.\n2.  **Early Tech Successes Pre-Tesla/SpaceX:** Before his global fame with Tesla and SpaceX, Musk had already built and sold two highly successful tech companies: Zip2 (co-founded 1995, sold 1999) and X.com, which evolved into PayPal (co-founded 1999, acquired by eBay 2002). This established his entrepreneurial prowess and generated significant wealth long before his current iconic ventures.", additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'grounding_metadata': {}, 'model_provider': 'google_genai'}, id='lc_run--74982b26-9a08-467b-9e09-a87b5b2411b5-0', usage_metadata={'input_tokens': 258, 'output_tokens': 1711, 'total_tokens': 1969, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 1420}})

#AI message is a simple class on what LLM retursn to us -- present in content --- "Here's the information about Elon Musk:\n\n### 1. Short Summary\n\nElon Musk is a South African-born entrepreneur and businessman, currently the wealthiest person in the world with an estimated net worth of $500 billion as of October 2025. He is renowned for leading major companies like Tesla, SpaceX, Twitter, and xAI. After co-founding early successful ventures such as Zip2 and PayPal, he became an American citizen in 2002, having previously held Canadian citizenship by birth.\n\n### 2. Two Very Interesting Things\n\n1.  **Complex Citizenship Journey:** Despite being born in South Africa, he obtained Canadian citizenship at birth through his Canadian-born mother and emigrated to Canada in 1989, before eventually becoming an American citizen in 2002. This means he's held citizenship from at least two, potentially three, different countries over his lifetime.\n2.  **Early Tech Successes Pre-Tesla/SpaceX:** Before his global fame with Tesla and SpaceX, Musk had already built and sold two highly successful tech companies: Zip2 (co-founded 1995, sold 1999) and X.com, which evolved into PayPal (co-founded 1999, acquired by eBay 2002). This established his entrepreneurial prowess and generated significant wealth long before his current iconic ventures."
# Also contains info about tokens consumed during the call  --- check messages video

# Also, check the response metadata --- model used, finish reason, # of tokens etc./.. all of this helps in debugging