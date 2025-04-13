from agents import Agent, function_tool
from extractive_summarizer_agent import create_abstractive_agent
from extractive_summarizer import ExtractiveSummarizer
from summary_output import SummaryOutput

def extractive_summary(feedback: str) -> str:
    extractive = ExtractiveSummarizer()
    return extractive.summarize(feedback)

def create_extractor_agent() -> Agent:
    extractive_tool = function_tool(
        func=extractive_summary,
        name_override="extractive_summarizer", 
        description_override="Extracts the sentences from reviews as a deterministic summary."
    )

    abstractive_agent = create_abstractive_agent()
    
    abstractive_tool = abstractive_agent.as_tool(
        tool_name="extractive_summarizer",
        tool_description="Extracts the sentences from reviews as a deterministic summary."
    )

    extractor_agent = Agent(
        name="Comparison Judge",
        instructions=(
            "You are an extractor agent that creates and compares different types of summaries."
            "Use the extractive and abstractive tools to create summaries of the feedback."
        ),
        tools=[extractive_tool, abstractive_tool],
        output_type=SummaryOutput
    )

    return extractor_agent

