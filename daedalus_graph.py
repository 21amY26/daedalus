from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent,RunContext
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List,Annotated, Any
from langgraph.config import get_stream_writer
from langgraph.types import interrupt
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import Client
import logfire
import os

#message classes from pydantic_ai.messages
from pydantic_ai.messages import(
    ModelMessage,
    ModelMessagesTypeAdapter
)

from pydantic_ai_coder import pydantic_ai_coder, PydanticAIDeps, list_documentation_pages_helper

#loading environment variables
load_dotenv()

#logfire config to suppress warning
logfire.configure(send_to_logfire='never')

base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
api_key = os.getenv('LLM_API_KEY', 'no-llm-api-key-provided')
is_ollama = "localhost" in base_url.lower()

# Initialize OpenAI client
if is_ollama:
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
supabase: Client=Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

#defining agest state schema
class AgentState(TypedDict):
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x,y : x + y]
    scope:str

#AGENT MODEL CREATION
reasoner_model = os.getenv('REASONER_MODEL','mistral:7b-instruct')
reasoner = Agent(
    OpenAIModel(
        reasoner_model,
        base_url=base_url,
        api_key=api_key
    ),
    system_prompt="You are an expert at engineering AI agents with deep knowledge of Pydantic AI and creating a detailed scope document including architecture diagrams, core components, dependencies, and testing strategy. Base your answer on user requests and available documentation."
)

coder_model=os.getenv('PRIMARY_MODEL','llama3.1:8b')
router_agent=Agent(
    OpenAIModel(
        coder_model,
        base_url=base_url,
        api_key=api_key,
    ),
    system_prompt="Your job is to decide whether user wants to 'finish_conversation' or continue 'coder_agent' based on their message."
)

end_convo_agent=Agent(
    OpenAIModel(
        coder_model,
        base_url=base_url,
        api_key=api_key
    ),
    system_prompt="Your job is to end the covnersation of creating an AI agent providing execution instruction of the agent and then wrap it with a polite goodbye."
)

#NODE DEFINITIONS
#scope definition node with reasoner llm
async def reasoner_defines_scope(state: AgentState):
    documentation_pages = await list_documentation_pages_helper(supabase)
    documentation_pages_str = "\n".join(documentation_pages)

    #definine scope using reasoner
    prompt=f"""
User AI Agent Request: {state['latest_user_message']}

Create a detailed scope document for the AI ganet following the give guidelines:
- Architecture diagram
- Core components
- External dependencies
- Testing Strategy

Consider the available documentation pages as well:
{documentation_pages_str}

Include a relevant list of documentation pages that can be used in building the agent.
"""
    result= await reasoner.run(prompt)
    scope=result.data

    #save scope to file
    os.makedirs("workbench",exist_ok=True)
    with open("workbench/scope.md","w",encoding="utf-8") as f:
        f.write(scope)

    return {"scope":scope}

#node for coder agent
async def coder_agent(state: AgentState, writer):
    deps=PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client,
        reasoner_output=state['scope']
    )
    message_history=[]
    for row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(row))

    prompt=f"""
User Feedback: {state['latest_user_message']}
Previous Feedback Summary: {state.get('feedback_summary','')}
Respond by coding/refining/clarifying based on prior output and feedback.
"""
    #run agent in a stream
    if is_ollama:
        writer=get_stream_writer()
        result=await pydantic_ai_coder.run(
            state['latest_user_message'],
            deps=deps,
            message_history=message_history,
        )
        writer(result.data)
    else:
        async with pydantic_ai_coder.run(
            state['latest_user_message'],
            deps=deps,
            message_history=message_history
        ) as result:
            async for chunk in result.stream_text(delta=True):
                writer(chunk)

    return{
        "messages":message_history+[result.new_messages_json()],
        "latest_user_message": state['latest_user_message']
    }

#node for graph interrupt to get next user message
def get_next_user_message(state:AgentState):
    return{"latest_user_message": interrupt({})}

#node for router decision: is agent built satifactorily?
async def route_user_message(state:AgentState):
    prompt=f'''
User sent the message:
{state['latest_user_message']}
Respond *only* with:
-"finish_conversation" if user wants to end conversation
-"coder_agent" if user wants to continue building agent
'''
    result= await router_agent.run(prompt)
    choice=result.data.strip() #returns 'finish_conversation' or coder_agent'

    if choice not in ("finish_conversation","coder_agent"):
        choice="coder_agent"
    
    return choice


#node for ending covnersation
async def finish_conversation(state: AgentState,writer):
    message_history=[]
    for row in state['messages']:
        message_history.extend(
        ModelMessagesTypeAdapter.validate.json(row))
    #running agent in a stream
    if is_ollama:
        writer=get_stream_writer()
        result= await end_convo_agent.run(
            state['latest_user_message'],
            message_history=message_history
        )
        writer(result.data)
    else:
        async with end_convo_agent.run(
            state['latest_user_message'],
            message_history=message_history
        ) as result:
            async for chunk in result.stream_text(delta=True):
                writer(chunk)
    
    return {"messages": [result.new_messages_json()]}

#build graph workflow
#let bob be the builder of the agentic workflow
bob=StateGraph(AgentState)
#adding nodes to the graph
bob.add_node("reasoner_defines_scope",reasoner_defines_scope)
bob.add_node("coder_agent",coder_agent)
bob.add_node("get_next_user_message", get_next_user_message)
bob.add_node("route_user_message", route_user_message)
bob.add_node("finish_conversation", finish_conversation)
#adding edges to the graph
bob.add_edge(START,"reasoner_defines_scope")
bob.add_edge("reasoner_defines_scope","coder_agent")
bob.add_edge("coder_agent","get_next_user_message")
bob.add_conditional_edges(
    "get_next_user_message",
    route_user_message,
    {"coder_agent":"coder_agent","finish_conversation":"finish_conversation"}
)
bob.add_edge("finish_conversation",END)


#MEMORY SAVER
memory=MemorySaver()
agentic_flow=bob.compile(checkpointer=memory)