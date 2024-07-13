from langchain_community.document_loaders import WebBaseLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain_anthropic import ChatAnthropic
import json
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict
from langgraph.graph import StateGraph, END

raw_model = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")

class Info(BaseModel):
    language: str
    prospect: Literal["Public Company", "Large Private Company", "AI Native Early Legit Startup", "Early Legit Startup", "Early Startup", "AI Native Early Startup", "AI Native Pre-Company", "Other", "Personal", "Consultancy", "Education"]
    horizontal: bool
    interest_reasoning: str
    interest_score: int


model = raw_model.with_structured_output(Info)

prompt = """You are doing research on companies. All these companies signed up for LangChain waitlist. \
LangChain is an LLMOps company. Your job is to fill out information about these companies that will help \
determine which ones are interesting to look at.

The information you should be gathering is:

Language: The language of the website, should be ISO 639 three-letter (set 2) codes (eng, fra, etc)

Horizontal: Whether this is a horizontal AI company or not. Horizontal AI companies include low-code platforms for building \
generic LLM applications

Prospect: What type of company this is. This includes:
- Public Company: if they are a publicly traded company
- Large Private Company: greater than 1 bil valuation
- AI Native Early Legit Startup: A legit startup (top tier investors) that is built around AI
- Early Legit Startup: A legit startup (top tier investors) that is NOT built around AI
- AI Native Early Startup: A early stage startup that does not have top tier VCs or real revenue but has some investment/revenue
- Early Startup: A early stage startup that does not have top tier VCs or real revenue but has some investment/revenue
- AI Native Pre-Company: A super early startup, or one that does not have any investment from good VCs or revenue
- Personal: personal websites
- Consultancy
- Education
- Other: doesn't fit any of the above categories

Interest: How interesting overall this company seems for LangChain to engage. Companies that are interesting are generally based \
close to LangChain (which is in San Francisco), hot, fast moving vertical startups or large private companies. \
However, if a "logo" would be good to \
have on LangChain website, then that is also interesting. The more information they provide in the `Info` field, \
that likely means they are more engaged, and therefor it is more interesting. \
If the company is horizontal, or seems to be consulting, they are less interesting. There are two fields here to fill out:
- Interest Reasoning: Reasoning for why or why not this company is interesting
- Interest Score: Score between 1-10 for how interesting this company is, 10 being more interesting

When filling out the form, they specified some information about how they hope to use the product. \
This was an optional field. The information they specified was:

<info>
{info}
</info>

Based on the below website, fill out information about the company {company_name}.

{website}"""

class State(TypedDict):
    url: str
    company_name: str
    info: str
    website: str
    response: Info
    new_url: str


def load_website(state: State):
    try:
        loader = WebBaseLoader("https://" + state['url'])
        docs = loader.load()
        return {"website": docs[0].page_content}
    except:
        pass

tool = TavilySearchResults()

class NewUrl(BaseModel):
    url: str

class Nothing(BaseModel):
    nothing: bool
    
def research_website(state: State):
    results = TavilySearchResults().run(state["company_name"])
    
    prompt = """You are trying to find the correct url for a given company. Here is the company info:
    
    name: {company_name}
    domain: {url}
    
    The domain was not found, so we think it is wrong. We searched for the company name, and got the following results:
    
    {results}
    
    Based on these results, either (a) try a new URL, or (b) decide that there is nothing present and give up. \
    If generating the URL, do not include the prefix (eg return `foo.com` not `https://foo.com`"""
    
    response = raw_model.bind_tools([NewUrl, Nothing]).invoke(prompt.format(
        results = results,
        company_name= state["company_name"],
        url=state["url"]
    ))
    if len(response.tool_calls) == 1:
        if response.tool_calls[0]['name'] == 'NewUrl':
            return {"new_url": response.tool_calls[0]['args']['url']}


def get_info(state: State):
    p = prompt.format(company_name=state['company_name'], website=state['website'], info=state['info'])
    result = model.invoke(p)
    return {"response": result}


def initial_success(state: State):
    if "website" not in state or not state['website']:
        return "research_website"
    else:
        return "get_info"

def load_new_website(state: State):
    try:
        loader = WebBaseLoader("https://" + state['new_url'])
        docs = loader.load()
        return {"website": docs[0].page_content}
    except:
        pass

def found_website(state: State):
    if "new_url" in state:
        return "load_new_website"
    else:
        return END


graph = StateGraph(State)
graph.add_node(load_website)
graph.add_node(get_info)
graph.add_node(load_new_website)
graph.add_node(research_website)
graph.add_conditional_edges("load_website", initial_success)
graph.add_conditional_edges("research_website", found_website)
graph.add_edge("load_new_website", "get_info")
graph.add_edge("get_info", END)
graph.set_entry_point("load_website")
graph = graph.compile()
