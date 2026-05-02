from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query : str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles , URLs and snippits"""
    try:
        response = tavily.search(query=query, max_results=5)
        out = []
        for r in response['results']:
            out.append(
                f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
            )
        return "\n---\n".join(out)
    except Exception as e:
        return f"Could not search web: {str(e)}"
    
@tool
def scrape_url(url : str) -> str:
    """Scrape and return clean text content from a given URL for deeper insights"""
    try:
        response = requests.get(url, timeout=8, headers={"User-Agent" : "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["scripts", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
    
print(scrape_url.invoke("https://www.hindustantimes.com/india-news/bengal-election-2026-live-updates-voting-percentage-magrahat-paschim-diamond-harbour-falta-tmc-mamata-bjp-eci-evm-booths-101777682455033.html"))