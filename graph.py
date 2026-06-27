import os, getpass
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import (HumanMessage, AIMessage, SystemMessage, BaseMessage)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from hybrid import hybrid_search
from reranker import rerank
from logger import logger

load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your GEMINI AI API key: ")

WINDOW_TURNS = 5
WINDOW_MESSAGES = WINDOW_TURNS * 2

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    retrieved_context: str
    sources: list[str]
    route_decision: str

def retrieve_node(state: ChatState) -> dict:
    """Run hybrid search + rerank on the current question."""
    try:
        logger.info(f"Retrieving Context from retrieve_node() - Query: {state['question']}")
        question = state["question"]
        candidates = hybrid_search(question, top_k=20)
        logger.info(f"Retrieved Context: {candidates}")
        top_hits = rerank(question, candidates, top_k=5)
        logger.info(f"Reranked Context: {top_hits}")
        context = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(top_hits))
        sources = list({h["source"] for h in top_hits})

        return {"retrieved_context": context,
                "sources": sources}
    except Exception as e:
        logger.error(f"Error in generate_node: {e}")


SYSTEM_PROMPT = """You are a helpful assistant answering questions using the \
provided context. Use the context as your primary source. If it partially \
addresses the question, answer with what it supports and note what's missing. \
Only say the information isn't available if the context is unrelated to the question.

You also have access to the recent conversation history — use it to resolve \
references like "that", "the second one", or follow-up questions."""

ROUTER_PROMPT = """You are a routing classifier. Given the conversation history and \
the user's new question, decide whether answering requires looking up new information \
from a document database, or whether the question can be answered using only the \
conversation history.

Reply with exactly one word:
- "retrieve" — if the question asks about new topics, or details not yet discussed
- "direct" — if the question is conversational, refers to \
something already in the recent assistant messages, or is a repeat of a prior question
"""

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


def router_node(state: ChatState) -> dict:
    """Decide whether to retrieve or skip straight to answering."""
    history = state["messages"][-WINDOW_TURNS:]

    if not history:
        return {"route_decision": "retrieve"}

    messages: list[BaseMessage] = [
        SystemMessage(content=ROUTER_PROMPT),
        *history,
        HumanMessage(content=f"New question: {state['question']}")
    ]

    logger.info("Response retured from previous chat history : router_node()")

    response = model.invoke(messages)
    decision = response.content.strip().lower()

    if decision not in ("retrieve", "direct"):
        decision = "retrieve"
    return {"route_decision": decision}

def route_condition(state:ChatState) -> str:
    """Conditional edge function — returns the name of the next node."""
    return "retrieve" if state["route_decision"] == "retrieve" else "generate"
    

def generate_node(state: ChatState) -> dict:
    """Build the prompt (system + windowed history + current Q+context),
    call the LLM, return new messages to append to state."""
    try:
        history= state["messages"][-WINDOW_MESSAGES:]
        context = state.get("retrieved_context", "")

        if context:
            current_user_content = f"Context:\n{context}\n\nQuestion: {state['question']}"
        else:
            current_user_content = state["question"]

        messages: list[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT),
            *history,
            HumanMessage(content=current_user_content)
        ]

        logger.info(f"Response returned from context and generated from generate_node.")

        response = model.invoke(messages)

        answer_text = response.content

        return {
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=answer_text)
            ]
        }
    except Exception as e:
        logger.error(f"Error in generate_node: {e}")

def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("route", router_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)


    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        route_condition,
        {"retrieve": "retrieve", "generate": "generate"} 
    )
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)

graph = build_graph()


def answer_question(question: str, thread_id:str) -> dict:
     """Public entrypoint — invoked by the FastAPI handler."""
     result = graph.invoke(
        {"question": question},
        config={"configurable": {"thread_id": thread_id}}
     )
     return {
        "answer": result["messages"][-1].content,
        "sources": result.get("sources", []),
        "thread_id": thread_id,
        "history_length": len(result["messages"])
     }