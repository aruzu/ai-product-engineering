import os
import asyncio
import csv
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from agents import function_tool
from .slack_agent import send_slack_message
from .time_agent import get_current_time
from firecrawl import FirecrawlApp, ScrapeOptions
#import requests
#import json

load_dotenv()

LLM_MODEL = "gpt-4o-mini"
client = OpenAI()
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

input_path = os.path.join(os.path.dirname(__file__), "..", "data", "viber_classified.csv")
output_path = os.path.join(os.path.dirname(__file__), "..", "data", "feature_research.md")

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

async def search_web(query: str) -> list:
    """Search the web using Firecrawl's search API"""
    try:
        search_results = firecrawl.search(query, limit=2)
        print("\n\n Search results: ", search_results)
        # Extract just the data part from the search results
        if hasattr(search_results, 'data'):
            return search_results.data
        return []
    except Exception as e:
        print(f" Error searching web: {str(e)}")
        return []

async def scrape_url(url: str) -> dict:
    """Scrape a URL using Firecrawl's scrape API"""
    try:
        print(f"\n\n Attempting to scrape URL: {url}")
        scrape_result = firecrawl.scrape_url(
            url,
            formats=['markdown', 'html'],
            only_main_content=True
        )
        print("\n\n Scrape result: ", scrape_result)
        if hasattr(scrape_result, 'data'):
            return scrape_result.data
        return {}
    except Exception as e:
        print(f" Error scraping URL: {str(e)}")
        return {}

def get_major_competitors(request):
    prompt = (
        "List the major competitors for Viber relevant to the following feature request. "
        "Return a numbered list of company/product names only.\n"
        f"Feature Request: {request}\n\nCompetitors:"
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0
    )
    return response.choices[0].message.content.strip()

def select_top_competitors(competitors, request):
    prompt = (
        "Given the following list of competitors and a feature request, select the 3 most relevant competitors for analysis. "
        "Return only their names as a comma-separated list.\n"
        f"Feature Request: {request}\n"
        f"Competitors: {competitors}\n\nTop 3 Competitors:"
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0
    )
    return response.choices[0].message.content.strip()

# --- LLM-as-a-Planner Pattern ---
# The LLM generates a step-by-step research plan before executing research and report generation.
async def generate_research_plan(request):
    # First, search the web for relevant information
    search_results = await search_web(f"Viber {request} feature competitors")
    
    # Create a context from search results
    search_context = "\n".join([f"- {result['title']}: {result['description']}" for result in search_results])
    
    prompt = (
        "You are a product research planner. Given the following feature request and web search results, "
        "outline a step-by-step research plan to investigate competitor products and user needs.\n"
        f"Feature Request: {request}\n"
        f"Web Search Results:\n{search_context}\n\nResearch Plan:"
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0
    )
    return response.choices[0].message.content.strip()

async def generate_feature_research(request, plan, top_competitors):
    # Search for each competitor's implementation of the feature
    competitor_research = []
    for competitor in top_competitors.split(", "):
        search_results = await search_web(f"{competitor} {request} feature implementation")
        for result in search_results:
            # Scrape the URL for detailed information
            scraped_content = await scrape_url(result["url"])
            if scraped_content:
                competitor_research.append(f"### {competitor}\n{scraped_content.get('markdown', '')}\n")
    
    competitor_analysis = "\n".join(competitor_research)
    
    prompt = (
        "You are a product research expert. Using the following research plan, competitor analysis, and web research, "
        "analyze the feature request and provide a structured research report.\n"
        f"Feature Request: {request}\n"
        f"Research Plan: {plan}\n"
        f"Top Competitors: {top_competitors}\n"
        f"Competitor Analysis:\n{competitor_analysis}\n\n"
        "Include the following sections:\n"
        "1. Market Analysis (with competitor comparison)\n"
        "2. User Impact\n"
        "3. Technical Feasibility\n"
        "4. Implementation Recommendations\n"
        "Be specific and cite competitor features where possible.\n\nResearch Report:"
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        web_search_options=(),
        max_tokens=1200,
        temperature=0
    )
    return response.choices[0].message.content.strip()

async def save_research_to_markdown(request, research):
    filepath = output_path
    print(f"\n\n Saving research to: {filepath}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    timestamp_result = await get_current_time()
    timestamp = getattr(timestamp_result, "final_output", str(timestamp_result))
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Feature Research Report\n\n")
            f.write(f"## Research Timestamp: {timestamp}\n\n")
            f.write(f"## Original Request\n\n{request}\n\n")
            f.write(f"## Research Analysis\n\n{research}\n")
        print(f"Successfully saved research to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving research to file: {str(e)}")
        return None

async def _send_slack_message_impl(message: str) -> str:
    try:
        # 1. Generate research plan with web search
        plan = await generate_research_plan(message)
        print("\n\n Generated plan:", plan)

        # 2. Get major competitors
        competitors = get_major_competitors(message)
        print("\n\n Found competitors:", competitors)

        # 3. Select top 3 competitors
        top_competitors = select_top_competitors(competitors, message)
        print("\n\n Selected top competitors:", top_competitors)
       
        # 4. Generate research report using plan, competitors, and web research
        research = await generate_feature_research(message, plan, top_competitors)
        print("\n\n Generated research:", research)

        # 5. Save to markdown
        filepath = await save_research_to_markdown(message, research)
        if not filepath:
            print("Failed to save research to file")
            return "failed"

        # 6. Send to Slack via MCP-based agent
        # status = await send_slack_message(research)
        # print("Status: Message sent to Slack")

        return "success"
    except Exception as e:
        print(f"Status: Failed to process request - {str(e)}")
        return "failed"

@function_tool
async def send_feature_research_to_slack(message: str) -> str:
    return await _send_slack_message_impl(message)

async def main():
    with open(input_path, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        feature_requests = [row for _, row in zip(range(3), reader)]
        print(f"Processing {len(feature_requests)} feature requests...")
        for index, row in enumerate(feature_requests, 1):
            if row.get("classification") != "feature request":
                continue
            request = row["content"]
            status = await _send_slack_message_impl(request)
            print(f"Feature request sent to Slack. Status: {status}")
    print(f"Feature research complete. Research reports saved in: {output_path}")

if __name__ == "__main__":
    asyncio.run(main()) 