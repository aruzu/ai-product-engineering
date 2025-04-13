from agents import Agent

def create_abstractive_agent() -> Agent:
    return Agent(
        name="Abstractive Summarizer",
        instructions="You briefly rephrase the essence of the feedback in your own words, focusing on the key ideas."
    )