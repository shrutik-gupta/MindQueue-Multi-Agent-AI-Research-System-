# Output of create_agent
# {
#     "messages" : [
#         HumanMessage(content="What is the capital of France?"),
#         AIMessage(content="", tool_calls=[{"name":"web_search",...}]),
#         ToolMessage(content="Title: Capital of France - Wikipedia..."),
#         AIMessage(content="The capital of France is Paris")
#     ]
# }

from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain


def _extract_tool_output(messages):
    tool_texts = []
    for msg in messages:
        if isinstance(msg, dict):
            msg_type = msg.get("type")
            content = msg.get("content")
        else:
            msg_type = getattr(msg, "type", None)
            content = getattr(msg, "content", None)
        if msg_type == "tool" or getattr(msg, "__class__", None).__name__ == "ToolMessage":
            if content:
                tool_texts.append(content)
    return "\n\n".join(tool_texts).strip()


def run_research_pipeline(topic : str) -> dict:
    state = {}
    
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [
            ("user", f"Find recent, reliable and detailed information about {topic}")
        ]
    })
    search_tool_output = _extract_tool_output(search_result["messages"])
    state["search_results"] = search_tool_output or search_result["messages"][-1].content
    print(f'\n\n{state["search_results"]}\n\n')

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user", 
            f"Based on the following search results about '{topic}',"
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })
    reader_tool_output = _extract_tool_output(reader_result["messages"])
    state["scraped_content"] = reader_tool_output or reader_result["messages"][-1].content
    print(f'\n\n{state["scraped_content"]}\n\n')

    research_combined = (
        f"Search results :\n{state['search_results']}\n\n"
        f"Detailed scraped content:\n{state['scraped_content']}"
    )
    state["report"] = writer_chain.invoke({
        "topic":topic,
        "research":research_combined
    })
    print(f'\n\n{state["report"]}\n\n')

    state["feedback"] = critic_chain.invoke({
        "report":state['report']
    })
    print(f'\n\n{state["feedback"]}\n\n')

    return state

if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)