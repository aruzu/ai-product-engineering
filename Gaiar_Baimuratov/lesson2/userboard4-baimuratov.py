# interview_agents.py
import asyncio, os, json, sys
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from agents import Agent, Runner, trace
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.table import Table
from rich.markdown import Markdown
import re

# Create rich console
console = Console()

# ---------- summary schema ---------- #
class SummaryReport(BaseModel):
    market_perspective: str   # short paragraph
    go_or_no_go: str          # "GO" or "NO‑GO"
    rationale: List[str]      # bullet list (3‑6 items)

# Define a schema for persona sentiment
class PersonaSentiment(BaseModel):
    name: str
    sentiment: str
    key_points: List[str]
    summary: str

# Define a sentiment analysis report
class SentimentAnalysis(BaseModel):
    personas: List[PersonaSentiment]

# ---------- agent factories ---------- #
def make_persona_agent(name: str, description: str) -> Agent:
    instructions = f"""
You are **{name}**.
Persona details: {description}

You're in a live group interview. For each question:
1. State your answer.
2. React to what others said (name them).
3. Explain *why* you agree, disagree, or find a point interesting—give a bit of personal context.
Limit to ≤120 words.
"""
    return Agent(
        name=name,
        instructions=instructions,
        model="gpt-4o"           # high‑quality reasoning
    )

def make_facilitator_agent(topic: str,
                           core_questions: List[str],
                           max_followups: int) -> Agent:
    instr = f"""
You are the *facilitator*.

### Topic
"{topic}"

### Duties per turn
1. Pick the next question:
   • Ask remaining core questions in order → {core_questions}
   • Then ≤{max_followups} follow‑ups that dig deeper into something you just heard.
2. Stop when done.

### Output (JSON)
{{"next_question": "<string>", "should_end": true|false}}
"""
    class FacOut(BaseModel):
        next_question: str
        should_end: bool

    return Agent(
        name="Facilitator",
        instructions=instr,
        output_type=FacOut,
        #don't change the model
        model="o4-mini"          # reasoning model
    )

def make_summarizer_agent() -> Agent:
    instr = """
You are a senior product strategist.  
Given the full transcript of a multi‑persona interview:

1. Capture the overall *market perspective* in ≤100 words.
2. Decide "GO" or "NO‑GO" for the idea.
3. List 3–6 bullet‑point reasons referencing the personas' remarks.

Return **exactly** a JSON object matching the SummaryReport schema.
"""
    return Agent(
        name="Summarizer",
        instructions=instr,
        output_type=SummaryReport,
        #don't change the model
        model="o3"               # latest reasoning model
    )

def make_sentiment_agent() -> Agent:
    instr = """
You are a sentiment analysis expert.
Analyze the transcript of each persona's responses to determine if they are:
1. Positive (enthusiastic, excited, supportive)
2. Neutral (balanced, considering pros and cons)
3. Negative (concerned, hesitant, skeptical)

For each persona, provide:
1. Overall sentiment (POSITIVE, NEUTRAL, or NEGATIVE)
2. Key points they made (2-3 bullet points)
3. A one-sentence summary of their perspective

Return a JSON structured as:
{
  "personas": [
    {
      "name": "Persona Name",
      "sentiment": "POSITIVE/NEUTRAL/NEGATIVE",
      "key_points": ["Point 1", "Point 2", "Point 3"],
      "summary": "One sentence summary"
    }
  ]
}
"""
    return Agent(
        name="Sentiment Analyzer",
        instructions=instr,
        output_type=SentimentAnalysis,
        #don't change the model
        model="gpt-4o"  # cost-effective for analysis
    )

# Helper function to convert transcript to a string message for facilitator
def transcript_to_facilitator_prompt(transcript, core_questions, asked_questions):
    prompt = f"""You are facilitating an interview about a smart water bottle product.
    
Questions to ask:
- Core questions: {core_questions}
- Already asked: {asked_questions}

Recent conversation:
"""
    
    # Add the most recent exchanges (most recent 5-10 exchanges)
    recent_msgs = transcript[-20:] if len(transcript) > 20 else transcript
    
    for msg in recent_msgs:
        role = msg.get("role", "")
        name = msg.get("name", "")
        content = msg.get("content", "")
        
        if role == "user":
            prompt += f"\nFacilitator: {content}\n"
        elif role == "assistant":
            if name:
                prompt += f"{name}: {content}\n"
            else:
                prompt += f"Assistant: {content}\n"
    
    prompt += "\nDecide on the next question to ask or if the interview should end."
    
    return [
        {"role": "system", "content": "You are a facilitator conducting an interview."},
        {"role": "user", "content": prompt}
    ]

# Helper function to convert transcript to a string message for personas
def transcript_to_persona_prompt(persona_name, persona_description, current_question, transcript):
    prompt = f"""You are {persona_name}. 
Details: {persona_description}

Current question: {current_question}

Here are some other responses to this question:
"""
    
    # Find responses to the current question
    responses_for_current_question = []
    collecting_current_responses = False
    
    for msg in transcript:
        if msg.get("role") == "user" and msg.get("content") == current_question:
            collecting_current_responses = True
            continue
            
        if collecting_current_responses and msg.get("role") == "assistant" and "name" in msg:
            if msg.get("name") != persona_name:  # Don't include own response
                name = msg.get("name", "Unknown")
                content = msg.get("content", "")
                responses_for_current_question.append(f"{name}: {content}")
        
        # If we encounter a new question, stop collecting
        if collecting_current_responses and msg.get("role") == "user" and msg.get("content") != current_question:
            break
    
    # Add responses to the prompt
    for response in responses_for_current_question:
        prompt += f"\n{response}\n"
    
    prompt += "\nProvide your answer to the question, considering what others have said."
    
    return [
        {"role": "system", "content": "You are in a group interview."},
        {"role": "user", "content": prompt}
    ]

# Helper function to convert transcript to a single string message for summarizer
def transcript_to_string_message(transcript):
    transcript_text = "Interview Transcript:\n\n"
    
    for msg in transcript:
        role = msg.get("role", "unknown")
        name = msg.get("name", "")
        content = msg.get("content", "")
        
        if role == "user":
            transcript_text += f"Facilitator: {content}\n\n"
        elif role == "assistant":
            if name:
                transcript_text += f"{name}: {content}\n\n"
            else:
                transcript_text += f"Assistant: {content}\n\n"
    
    # Return a single message with the entire transcript as content
    return [
        {"role": "system", "content": "Analyze this interview transcript about a smart water bottle product."},
        {"role": "user", "content": transcript_text}
    ]

# Function to create a sentiment analysis prompt
def create_sentiment_prompt(transcript, personas):
    persona_responses = {}
    for p in personas:
        persona_responses[p["name"]] = []
    
    current_question = None
    
    for msg in transcript:
        if msg.get("role") == "user":
            current_question = msg.get("content", "")
        elif msg.get("role") == "assistant" and "name" in msg and msg.get("name") in persona_responses:
            name = msg.get("name")
            content = msg.get("content", "")
            persona_responses[name].append({
                "question": current_question,
                "response": content
            })
    
    # Create a prompt with each persona's responses
    prompt = "Analyze the sentiment and key points from each persona in this interview:\n\n"
    
    for name, responses in persona_responses.items():
        prompt += f"## {name}\n"
        for resp in responses:
            prompt += f"Question: {resp['question']}\n"
            prompt += f"Response: {resp['response']}\n\n"
    
    return [
        {"role": "system", "content": "You are a sentiment analysis expert."},
        {"role": "user", "content": prompt}
    ]

# Function to print colored persona responses
def print_persona_response(persona_name, response):
    # Different styles for different personas
    styles = {
        "Alice (Athlete)": {"color": "blue", "emoji": "🏃‍♀️"},
        "Bob (Office Worker)": {"color": "green", "emoji": "💼"},
        "Claire (Parent)": {"color": "magenta", "emoji": "👩‍👦"},
        "Facilitator": {"color": "yellow", "emoji": "🎯"}
    }
    
    # Get style or use default
    style = styles.get(persona_name, {"color": "white", "emoji": "💬"})
    
    # Highlight any mentions of other personas
    highlighted_text = response
    for other_name in styles.keys():
        if other_name in highlighted_text and other_name != persona_name:
            pattern = re.compile(f"({re.escape(other_name)})", re.IGNORECASE)
            highlighted_text = pattern.sub(r"[bold]\1[/bold]", highlighted_text)
    
    # Create panel with styled text
    panel_title = f"{style['emoji']} {persona_name}"
    
    # Format the response
    formatted_text = Text(highlighted_text)
    
    # Create and display panel
    panel = Panel(
        formatted_text,
        title=panel_title,
        border_style=style["color"],
        box=box.ROUNDED,
        expand=False,
        padding=(1, 2)
    )
    
    console.print(panel)

# Function to print facilitator questions
def print_facilitator_question(question):
    # Create styled panel for facilitator question
    panel = Panel(
        Text(question, style="white"),
        title="🎙️ Facilitator Asks",
        border_style="bright_yellow",
        box=box.DOUBLE,
        expand=False,
        padding=(1, 2)
    )
    
    console.print("\n")
    console.print(panel)
    console.print("\n")

# Function to print executive summary in a rich format
def print_executive_summary(report, sentiment_analysis):
    # Clear screen
    console.clear()
    
    # Title
    console.print("[bold white on blue]EXECUTIVE SUMMARY REPORT[/bold white on blue]", justify="center")
    console.print("\n")
    
    # Create Go/No-Go badge
    if report.go_or_no_go == "GO":
        decision_style = "bold white on green"
    else:
        decision_style = "bold white on red"
    
    decision_panel = Panel(
        f"[{decision_style}]{report.go_or_no_go}[/{decision_style}]",
        title="Decision",
        border_style="bright_white",
        box=box.ROUNDED,
        padding=(1, 2)
    )
    
    # Create market perspective panel
    market_panel = Panel(
        Markdown(report.market_perspective),
        title="Market Perspective",
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(1, 2)
    )
    
    # Render top panels in a side-by-side layout
    console.print(decision_panel)
    console.print(market_panel)
    
    # Create rationale table
    rationale_table = Table(
        title="Key Rationale",
        box=box.ROUNDED,
        border_style="bright_cyan",
        header_style="bold bright_white",
        expand=True
    )
    
    rationale_table.add_column("#", style="dim", width=3)
    rationale_table.add_column("Point", style="bright_white")
    
    for i, point in enumerate(report.rationale, 1):
        rationale_table.add_row(str(i), point)
    
    console.print(rationale_table)
    console.print("\n")
    
    # Create persona sentiment table
    sentiment_table = Table(
        title="Persona Sentiment Analysis",
        box=box.ROUNDED,
        border_style="bright_magenta",
        header_style="bold bright_white",
        expand=True
    )
    
    sentiment_table.add_column("Persona", style="bright_white")
    sentiment_table.add_column("Sentiment", style="bright_white")
    sentiment_table.add_column("Summary", style="bright_white")
    sentiment_table.add_column("Key Points", style="bright_white")
    
    for persona in sentiment_analysis.personas:
        # Access properties using dot notation for Pydantic models
        name = persona.name
        sentiment = persona.sentiment
        summary = persona.summary
        key_points = "\n".join([f"• {point}" for point in persona.key_points])
        
        # Determine sentiment color
        if sentiment == "POSITIVE":
            sentiment_style = "[bold green]POSITIVE[/bold green]"
        elif sentiment == "NEGATIVE":
            sentiment_style = "[bold red]NEGATIVE[/bold red]"
        else:
            sentiment_style = "[bold yellow]NEUTRAL[/bold yellow]"
            
        sentiment_table.add_row(name, sentiment_style, summary, key_points)
    
    console.print(sentiment_table)
    console.print("\n")
    
    # Footer
    console.print("[italic]Generated by AI Product Research Suite[/italic]", justify="center")
    console.print(f"[dim]Report Date: {os.popen('date').read().strip()}[/dim]", justify="center")

# ---------- orchestration loop ---------- #
async def run_interview(topic: str,
                        personas: List[Dict[str, str]],
                        core_questions: List[str],
                        max_followups: int = 2):

    facilitator = make_facilitator_agent(topic, core_questions, max_followups)
    persona_agents = [make_persona_agent(p["name"], p["description"])
                      for p in personas]
    summarizer = make_summarizer_agent()
    sentiment_agent = make_sentiment_agent()

    transcript = []
    followups_used = 0
    asked_questions = []
    
    # Display interview header
    console.print("\n")
    console.print(f"[bold white on blue]PRODUCT RESEARCH INTERVIEW[/bold white on blue]", justify="center")
    console.print(f"[bold]Topic:[/bold] {topic}", justify="center")
    console.print("\n")
    
    # Display personas
    persona_table = Table(title="Interview Participants", box=box.ROUNDED)
    persona_table.add_column("Name", style="bold")
    persona_table.add_column("Description")
    
    for p in personas:
        persona_table.add_row(p["name"], p["description"])
    
    console.print(persona_table)
    console.print("\n")
    
    async def ask_persona(agent, persona_name, persona_description, current_question, transcript_so_far):
        # Convert to string-based prompt
        prompt = transcript_to_persona_prompt(persona_name, persona_description, current_question, transcript_so_far)
        return await Runner.run(agent, prompt)

    with trace("Interview run"):
        console.print("[bold]Starting Interview...[/bold]")
        console.print("[dim]" + "―" * 80 + "[/dim]")
        
        while True:
            # Convert transcript to string-based prompt for facilitator
            fac_prompt = transcript_to_facilitator_prompt(transcript, core_questions, asked_questions)
            
            # Run facilitator with string-based prompt
            fac_run = await Runner.run(facilitator, fac_prompt)
            fac_out = fac_run.final_output
            
            # Check if we should end
            if fac_out.should_end:
                console.print("\n[bold yellow]Facilitator:[/bold yellow] That's all – thanks everyone!\n")
                break
                
            # Record the question
            question = {"role": "user", "content": fac_out.next_question}
            transcript.append(question)
            asked_questions.append(fac_out.next_question)
            
            # Print facilitator question
            print_facilitator_question(fac_out.next_question)
            
            # Ask each persona
            for i, agent in enumerate(persona_agents):
                persona = personas[i]
                persona_name = persona["name"]
                persona_description = persona["description"]
                
                # Run persona agent with string-based prompt
                run = await ask_persona(
                    agent, 
                    persona_name, 
                    persona_description,
                    fac_out.next_question,
                    transcript
                )
                
                # Record response
                response = {
                    "role": "assistant", 
                    "content": run.final_output,
                    "name": persona_name
                }
                transcript.append(response)
                
                # Print colored response
                print_persona_response(persona_name, run.final_output)
            
            # Track follow-ups
            if fac_out.next_question not in core_questions:
                followups_used += 1
            if followups_used >= max_followups or len(asked_questions) >= len(core_questions) + max_followups:
                break
        
        # ---------- final analysis ---------- #
        console.print("\n[bold]Generating Analysis...[/bold]")
        
        # Run sentiment analysis
        sentiment_prompt = create_sentiment_prompt(transcript, personas)
        sentiment_run = await Runner.run(sentiment_agent, sentiment_prompt)
        sentiment_results = sentiment_run.final_output
        
        # Generate summary
        string_messages = transcript_to_string_message(transcript)
        sum_run = await Runner.run(summarizer, string_messages)
        report = sum_run.final_output
        
        # Print executive summary with rich formatting
        print_executive_summary(report, sentiment_results)

# ---------- example invocation ---------- #
if __name__ == "__main__":
    #os.environ["OPENAI_API_KEY"] = "sk-..."   # put your key here
    load_dotenv()
    
    # Check if rich is installed
    try:
        import rich
    except ImportError:
        print("Installing rich package for text formatting...")
        os.system("pip install rich")
        print("Rich package installed. Restarting script...")
        os.execv(sys.executable, ['python'] + sys.argv)
    
    topic = "A subscription‑based smart water bottle that reminds users to drink."
    personas = [
        {"name": "Alice (Athlete)",
         "description": "26‑year‑old marathon runner; tracks hydration closely"},
        {"name": "Bob (Office Worker)",
         "description": "45‑year‑old desk worker; often forgets to drink water"},
        {"name": "Claire (Parent)",
         "description": "35‑year‑old parent juggling childcare and work"}
    ]
    core_qs = [
        "What is your initial reaction to the idea?",
        "Describe a situation where this bottle would help you.",
        "What concerns do you have about the subscription model?"
    ]

    asyncio.run(run_interview(topic, personas, core_qs, max_followups=3))