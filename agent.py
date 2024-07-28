from langchain_community.document_loaders import WebBaseLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain_anthropic import ChatAnthropic
import json
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage


search_tool = TavilySearchResults(name="Search")
raw_model = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")

class Info(BaseModel):
    headquarters: str = Field(description='City in which this company is headquartered, should be in the form "San Francisco, USA"')
    number_employees: Optional[int] = Field(description="number of full time employees")
    investors: str = Field(description="Notable investors, should be a comma separated list")
    capital_raised: str = Field(description="amount of money raised, should be like: `None`, `10k`, `35m`, `1b` etc")
    ceo: str = Field(description="Name of CEO")
    ceo_college: str = Field(description="Where did the CEO go to college")


main_prompt = """You are doing research on companies. You are trying to figure out this information:

<info>
{info}
</info>


You have access to the following tools:

- `Search`: call a search tool and get back some results
- `ScrapeWebsite`: scrape a website and get relevant notes about the company. This will update the notes above.
- `Info`: call this when you are done and have gathered all the relevant info

Here is the information you have about the company you are researching:

Company Name: {company_name}"""

info_prompt = """You are doing research on companies. You are trying to figure out this information:

<info>
{info}
</info>

You just scraped the following website: {url}

Based on the website content below, jot down some notes about the website.

{content}"""

checker_prompt = """I am thinking of calling the info tool with the info below. \
Is this good? Give your reasoning as well. \
You can encourage the Assistant to look at specific URLs if that seems relevant, or do more searches.
If you don't think it is good, you should be very specific about what could be improved.

{presumed_info}"""
def ScapeWebsite(url: str):
    """Used to scrape a website"""
    loader = WebBaseLoader(url)
    docs = loader.load()
    website = docs[0].page_content
    p = info_prompt.format(info=Info.schema_json(), url=url, content=website)
    response = raw_model.invoke(p)
    return response

class GraphState(MessagesState):
    company_name: str
    info: Info

class InputSchema(TypedDict):
    company_name: str

class OutputSchema(TypedDict):
    info: Info

class Good(BaseModel):
    reason: str
    good: bool

model = raw_model.bind_tools([ScapeWebsite, search_tool, Info])
def call_model(state):
    p = main_prompt.format(info=Info.schema_json(), company_name=state['company_name'])
    messages = [{"role": "human", "content": p}] + state['messages']
    return {"messages": model.invoke(messages)}


def call_checker(state):
    p = main_prompt.format(info=Info.schema_json(), company_name=state['company_name'])
    messages = [{"role": "human", "content": p}] + state['messages'][:-1] # get rid of the last one
    presumed_info = state['messages'][-1].tool_calls[0]['args']
    p1 = checker_prompt.format(presumed_info=presumed_info)
    messages.append({"role": "human", "content": p1})
    response = raw_model.with_structured_output(Good).invoke(messages)
    if response.good:
        try:
            return {"info": Info(**state['messages'][-1].tool_calls[0]['args'])}
        except Exception as e:
            return {"messages": [ToolMessage(tool_call_id=state['messages'][-1].tool_calls[0]['id'], content=f"Invalid response: {e}")]}
    else:
        return {"messages": [ToolMessage(tool_call_id=state['messages'][-1].tool_calls[0]['id'], content=str(response), artifact=response)]}

tool_node = ToolNode([search_tool, ScapeWebsite])

def bad_agent(state):
    return {"messages": [{"content": "You must call one, and only one, tool!", "role": "user"}]}

def route_after_agent(state) -> Literal["bad_agent", "call_checker", "tool_node"]:
    last_message = state['messages'][-1]
    if len(last_message.tool_calls) != 1:
        return "bad_agent"
    elif last_message.tool_calls[0]['name'] == "Info":
        return "call_checker"
    else:
        return "tool_node"


def route_after_checker(state) -> Literal[END, "call_model"]:
    if 'info' in state:
        return END
    return "call_model"

graph = StateGraph(GraphState, input=InputSchema, output=InputSchema)
graph.add_node(call_model)
graph.add_node(call_checker)
graph.add_node(bad_agent)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("call_model")
graph.add_conditional_edges("call_model", route_after_agent)
graph.add_edge("tool_node", "call_model")
graph.add_conditional_edges("call_checker", route_after_checker)
graph.add_edge("bad_agent", "call_model")
graph = graph.compile()

