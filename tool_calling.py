from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import HumanMessage
from openai import OpenAI
import os

# Load environment variables from .env file
load_dotenv()

# Load the API keys from .env
google_search = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))

def get_search_result(query: str):
    """This function search the internet via google"""
    return google_search.run(query)

def tool_calling(user_query: str):
    # There is deepseek-chat and deepseek-code models which is updated to the latest version
    MODEL = "deepseek-chat"
    model = ChatOpenAI(model='deepseek-chat', openai_api_key=os.getenv("DEEPSEEK_API_KEY"), openai_api_base='https://api.deepseek.com',max_tokens=10000)
    client= OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    message = HumanMessage(
        content=[{"type": "text", "text": f"You are a helpful assistant. Answer {user_query} accurately"}],
    )

    model_with_tools = model.bind_tools([get_search_result])
    response = model_with_tools.invoke([message]) 
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "get_search_result":
            print("serper is called")
            query = tool_call["args"]["query"]
            # Google search the query using serper API
            search_content = get_search_result(query)
            # Return the content back to llm for further processing
            followup_response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant that can seaarch content on google when needed."},
                    {"role": "user", "content": f"You are a helpful assistant. Answer {user_query} accurately"},
                    {"role": "assistant", "content": f"Tool {tool_call['name']} returned: {search_content}"}
               ]
            )
            return followup_response.choices[0].message.content

    else:            
        # If no function call, return initial response
        return response.content   
