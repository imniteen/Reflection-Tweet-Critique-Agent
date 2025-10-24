import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


# Helper function to get environment variables
def _get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. Please set it before running the agent."
        )
    return value


# Prompts for reflection and generation
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=_get_env_var("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    api_version=_get_env_var("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=_get_env_var("AZURE_OPENAI_ENDPOINT"),
    api_key=_get_env_var("AZURE_OPENAI_API_KEY")
)

# Create chains
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    
    # Get tweet content from user input
    print("Please enter the tweet you want to improve:")
    tweet_content = input().strip()
    
    if not tweet_content:
        print("No tweet content provided. Exiting.")
        exit(1)
    
    inputs = {
        "messages": [
            HumanMessage(
                content=f'Make this tweet better: "{tweet_content}"'
            )
        ]
    }
    response = graph.invoke(inputs)
    
    # Extract and display the final tweet in a readable format
    print("\n" + "="*50)
    print("FINAL TWEET:")
    print("="*50)
    
    # Get the last AI message which contains the final tweet
    messages = response["messages"]
    final_tweet = None
    
    # Find the last AI message
    for message in reversed(messages):
        if hasattr(message, 'content') and message.__class__.__name__ == 'AIMessage':
            final_tweet = message.content
            break
    
    if final_tweet:
        print(final_tweet)
    else:
        print("No final tweet found in response")
    
    print("\n" + "="*50)
    print("CONVERSATION HISTORY:")
    print("="*50)
    
    # Display the conversation flow
    for i, message in enumerate(messages, 1):
        message_type = "USER" if message.__class__.__name__ == 'HumanMessage' else "AI"
        print(f"\n[{i}] {message_type}:")
        print("-" * 30)
        print(message.content)
        print("-" * 30)