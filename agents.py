from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

#Setting up Mistral model
llm = ChatMistralAI(model = "mistral-small", temperature=0)

#Search agent that uses web_search tool for searching top URLs
def build_search_agent():
    return create_agent(
        model = llm,
        tools = [web_search],
        system_prompt="You MUST use the web_search tool to retrieve the latest titles, URLs, and snippets. Do not answer using only your internal knowledge."
    )

#Reader agent that uses scapre_url tool for fetching indepth content from a given URL
def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scrape_url],
        system_prompt="You MUST use the scrape_url tool to fetch real content. Do not answer without using it."
    )

# Prompt template for generating a research report
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.
    
    Topic: {topic}
    
    Research Gathered: 
    {research}

    Structure the report as:
    - Introduction
    - Key Findings (minimum 3 well-explained points)
    - Conclusion
    - Sources (list all URLs found in the research)

    Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# Prompt template for evaluating the generated report
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

    Report:
    {report}

    Respond in this exact format:

    Score: X/10

    Strengths:
    - ...
    - ...

    Areas to Improve:
    - ...
    - ...

    One line verdict:
    ..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
