"""
Competitive Analysis Agent

Researches the main competitors in a category and builds a messaging map:
positioning statements, messaging pillars, claims (verbatim + summary) and white space.

Pattern: ReAct loop (agent <-> tools), then one structured-output step.
Flow:   START -> agent <-> tools -> finalize -> END

Usage (from this folder, with .venv active):
    python competitive_analysis_agent.py "vector databases" --thread vdb-1    # run the agent (use a new thread each run)
    python competitive_analysis_agent.py --inspect vdb-1                       # replay a saved run (runs nothing)
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional, Union

import httpx                             # fallback page fetch
from bs4 import BeautifulSoup            # strips HTML in the fallback
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedStore
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.store.base import BaseStore

load_dotenv()

# ---------------------------------------------------------------
# SETTINGS: everything you might want to tune, in one place
MODEL_NAME = "gpt-6-luna"
REASONING_EFFORT = "none"
SEARCH_MAX_RESULTS = 5      # results per web_search call
MAX_PAGE_CHARS = 6000       # keep each page small so the context window doesn't blow up
RECURSION_LIMIT = 40        # max graph steps per run, a safety stop for the ReAct loop
HERE = Path(__file__).parent              # databases live next to this file, wherever you run it from
CHECKPOINT_DB = HERE / "checkpoints.db"   # thread history (pause/resume, inspection)
STORE_DB = HERE / "store.db"              # saved maps, read by get_previous_map

log = logging.getLogger("competitive_analysis")
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# ---------------------------------------------------------------
# 1. OUTPUT SCHEMA: what the finished map looks like
# Field descriptions are sent to the model, so they double as instructions.

class CompetitorCandidate(BaseModel):    # proposed in confirm_competitors, before research
    name: str = Field(description="Company or product name, e.g. 'Pinecone'")
    url: str = Field(description="Homepage URL, e.g. 'https://www.pinecone.io/'")

class Claim(BaseModel):
    messaging_pillar: str = Field(description="The messaging pillar this claim supports, e.g. 'speed', 'cost', 'ease of use'")
    claim_verbatim: str = Field(description="Exact sentence or phrase copied word for word from a fetched page")
    claim_summary: str = Field(description="One-line plain-English summary of what the claim means")
    source_url: str     # required: no source, no claim

class Competitor(BaseModel):
    name: str
    url: str
    positioning_statement: str = Field(description="Main headline copied word for word from the competitor's homepage")
    claims: List[Claim]

class MessagingPillar(BaseModel):
    pillar: str = Field(description="Messaging pillar name, e.g. 'speed'")
    competitors: List[str]           # who claims it

class WhiteSpace(BaseModel):
    angle: str          # e.g. "compliance for regulated industries"
    why_open: str       # which competitors ignore it

class MessagingMap(BaseModel):
    category: str
    competitors: List[Competitor]
    messaging_pillars: List[MessagingPillar]
    white_space: List[WhiteSpace]
    changes_since_last_run: List[str]


# ---------------------------------------------------------------
# 2. STATE: the shared scratchpad every node reads and writes

class AgentState(TypedDict):
    category: str
    messages: Annotated[list[AnyMessage], add_messages]   # agent's working history
    final_map: Optional[MessagingMap]


# ---------------------------------------------------------------
# 3. TOOLS: the actions the agent can choose to take

@tool
def web_search(query: str) -> list[dict]:
    """Search the web. Use to find the main players in a category."""
    # return only what the agent needs, not the raw Tavily payload
    response = tavily.search(query=query, max_results=SEARCH_MAX_RESULTS)
    return [
        {"title": r["title"], "url": r["url"], "snippet": r["content"][:300]}
        for r in response["results"]
    ]

@tool
def fetch_page(url: str) -> str:
    """Fetch a page's text: homepage, pricing, or 'why us' page.
    Use when a homepage is too vague to extract clear claims."""
    # try Tavily Extract first (handles JS-heavy marketing sites well)
    try:
        result = tavily.extract(urls=[url])
        if result["results"]:
            return result["results"][0]["raw_content"][:MAX_PAGE_CHARS]
        log.warning("Tavily extract returned no content for %s; falling back to plain HTTP", url)
    except Exception as e:
        log.warning("Tavily extract failed for %s (%s); falling back to plain HTTP", url, e)

    # fallback: plain HTTP fetch + strip HTML
    try:
        html = httpx.get(url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()                      # drop non-messaging noise
        text = " ".join(soup.get_text(" ").split())
        return text[:MAX_PAGE_CHARS]
    except Exception as e:
        # return the error as text so the agent can decide what to do next
        return f"Could not fetch {url}: {e}. Try a different page or skip."

@tool
def confirm_competitors(competitors: list[CompetitorCandidate]) -> str:
    """Show the proposed competitor list to the human before fetching pages.
    Call this once you have a candidate list."""
    # plain dicts in the pause payload: it is saved in the checkpoint and shown to the human
    proposed = [CompetitorCandidate.model_validate(c).model_dump() for c in competitors]
    edited = interrupt({"proposed": proposed,
                        "question": "Add or remove any competitors?"})
    edited = [CompetitorCandidate.model_validate(c) for c in edited]   # check the human's edit too
    # tell the agent plainly that confirmation is done, so it doesn't ask again
    names = ", ".join(c.name for c in edited)
    details = "; ".join(f"{c.name} ({c.url})" for c in edited)
    return (f"CONFIRMED by the human. Map ONLY these competitors: {names}. "
            f"Details: {details}. Do NOT call confirm_competitors again. "
            f"Next step: fetch their pages.")

@tool
def get_previous_map(category: str,
                     store: Annotated[BaseStore, InjectedStore()]) -> Union[dict, str]:
    """Load the last saved map for this category, to spot what changed."""
    # LangGraph injects `store`; the model never sees this argument
    item = store.get(namespace=("maps", category.lower()), key="latest")
    if item is None:
        return "No previous map for this category. This is the first run."
    return item.value       # the MessagingMap saved last time, as a dict

tools = [web_search, fetch_page, confirm_competitors, get_previous_map]


# ---------------------------------------------------------------
# 4. NODES: the steps in the graph

SYSTEM_PROMPT = """You map the messaging of a product category.
1. Find the 4-8 main players. 2. Confirm the list with the human ONCE (never call confirm_competitors twice).
3. Fetch up to 3 pages per competitor. 4. Extract claims with source URLs.
5. Check the previous map for changes. Stop when every competitor has claims."""

# one model, used two ways: tool calling in agent_node, structured output in finalize_node
model = ChatOpenAI(model=MODEL_NAME, reasoning_effort=REASONING_EFFORT)
agent_model = model.bind_tools(tools)    # tells the model which tools exist and their arguments

def agent_node(state: AgentState):
    """The ReAct brain: reads history, decides the next tool call or stops."""
    # system prompt + category + everything so far (tool calls and results)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT + f"\n\nCategory: {state['category']}"),
        *state["messages"],
    ]
    response = agent_model.invoke(messages)
    # response either contains tool_calls (keep looping) or plain text (done researching)
    return {"messages": [response]}      # add_messages appends it to the history

FINALIZE_PROMPT = """Using only the research in this conversation, build the messaging map.
- Use only claims that appeared in fetched pages, each with its source_url. Never invent claims.
- For each claim, copy the claim word for word from the page, then add a one-line summary in your own words.
- The positioning statement is the competitor's own homepage headline, copied word for word. Do not write your own.
- Group claims into 4-8 messaging pillars and list which competitors claim each pillar.
- White space = angles relevant to buyers that no competitor clearly claims.
- If a previous map was loaded, list concrete changes. Otherwise return an empty list."""

tool_node = ToolNode(tools)   # runs whatever tool the agent picked

def finalize_node(state: AgentState, *, store: BaseStore):
    """Turn the research into the structured MessagingMap and save it."""
    # structured output: the model must return a valid MessagingMap, not free text
    final_map = model.with_structured_output(MessagingMap).invoke([
        SystemMessage(content=FINALIZE_PROMPT),
        *state["messages"],              # all the research the agent gathered
    ])

    # save under two keys in the same namespace
    namespace = ("maps", state["category"].lower())
    data = final_map.model_dump()
    store.put(namespace, "latest", data)                          # what get_previous_map reads
    run_key = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
    store.put(namespace, run_key, data)                           # dated history of every run
    return {"final_map": final_map}


# ---------------------------------------------------------------
# 5. ROUTING: tool call -> run tools, otherwise -> finalize

def route_after_agent(state: AgentState):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else "finalize"


# ---------------------------------------------------------------
# 6. WIRING

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent,
                              {"tools": "tools", "finalize": "finalize"})
builder.add_edge("tools", "agent")        # the ReAct loop
builder.add_edge("finalize", END)

memory = SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False))
# saved maps persist across restarts; isolation_level=None (autocommit) is required,
# because the store manages its own transactions
store = SqliteStore(sqlite3.connect(STORE_DB, check_same_thread=False, isolation_level=None))
store.setup()                 # creates the tables on first run; safe to call again
graph = builder.compile(checkpointer=memory, store=store)


def run(category: str, thread_id: str):
    """Run the agent, handling the pause at confirm_competitors.
    Use a new thread_id for every run. To see what the agent did, call inspect_run(graph, thread_id)."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    result = graph.invoke(
        {"category": category, "messages": [HumanMessage(content=f"Map the messaging for: {category}")]},
        config,
    )

    # The graph paused inside confirm_competitors -> show the list, get the human's edit
    while "__interrupt__" in result:
        proposed = result["__interrupt__"][0].value["proposed"]
        print("Proposed competitors:")
        for i, c in enumerate(proposed):
            print(f"  {i}: {c['name']}: {c['url']}")
        drop = input("Indexes to remove (comma-separated, blank = keep all): ").strip()
        keep = [c for i, c in enumerate(proposed)
                if str(i) not in {d.strip() for d in drop.split(",") if d.strip()}]
        result = graph.invoke(Command(resume=keep), config)   # resume where it paused

    return result["final_map"]


# ---------------------------------------------------------------
# 7. DISPLAY: for people only. Other code uses the MessagingMap that run() returns.

def show(title, body=""):
    print(f"\n### {title}\n\n{body}")

def cell(text):
    # keep markdown tables intact: no pipes or line breaks inside a cell
    return str(text).replace("|", "/").replace("\n", " ")

def show_map(m):
    """Show a finished map as a readable report."""
    show(f"Messaging map: {m.category}")
    for c in m.competitors:
        rows = "\n".join(f"| {cell(cl.messaging_pillar)} | \"{cell(cl.claim_verbatim)}\" | {cell(cl.claim_summary)} | {cl.source_url} |"
                         for cl in c.claims)
        show(f"{c.name}",
             f"**Positioning statement:** {c.positioning_statement}\n\n"
             f"| Messaging pillar | Claim (Verbatim Copy) | Claim (Summary) | Source |\n|---|---|---|---|\n{rows}")
    rows = "\n".join(f"| {cell(p.pillar)} | {', '.join(p.competitors)} |" for p in m.messaging_pillars)
    show("Messaging pillars", f"| Messaging pillar | Claimed by |\n|---|---|\n{rows}")
    show("White space", "\n".join(f"- **{w.angle}**: {w.why_open}" for w in m.white_space) or "None found")
    show("Changes since last run", "\n".join(f"- {x}" for x in m.changes_since_last_run) or "None (or first run)")

def inspect_run(graph, thread_id):
    """Show step by step what the agent did in a past run.
    Reads the saved thread only: no new searches, no API calls."""
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    if not messages:
        print(f"No saved run for thread '{thread_id}'.")
        return

    for msg in messages:
        if isinstance(msg, HumanMessage):
            show("Request", msg.content)
        elif isinstance(msg, AIMessage) and msg.tool_calls:          # REASON
            calls = "\n".join(f"- `{t['name']}`({', '.join(f'{k}={str(v)[:80]}' for k, v in t['args'].items())})"
                              for t in msg.tool_calls)
            show("Agent decided to call", calls)
        elif isinstance(msg, AIMessage):
            show("Agent: done researching", msg.content[:500])
        elif isinstance(msg, ToolMessage):                           # ACT + OBSERVE
            show(f"Tool result: `{msg.name}`", f"> {cell(msg.content)[:300]}...")


# ---------------------------------------------------------------
# 8. COMMAND LINE: what happens when you run `python competitive_analysis_agent.py ...`

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Competitive analysis: map how competitors in a category position themselves.")
    parser.add_argument("category", nargs="?", help='product category, e.g. "vector databases"')
    parser.add_argument("--thread", help="thread ID for this run; use a new one every run")
    parser.add_argument("--inspect", metavar="THREAD", help="replay a saved run step by step (runs nothing)")
    args = parser.parse_args()

    if args.inspect:
        inspect_run(graph, args.inspect)
    elif args.category and args.thread:
        show_map(run(args.category, args.thread))
    else:
        parser.error('give a category and --thread (e.g. "vector databases" --thread vdb-1), or --inspect THREAD')
