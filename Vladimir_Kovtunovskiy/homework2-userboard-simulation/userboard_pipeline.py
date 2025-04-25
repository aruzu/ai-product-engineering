"""
phase3_multiagent.py
====================

Setup
-----
```bash
pip install langchain langgraph openai python-dotenv rich
export OPENAI_API_KEY="sk‑..."
python phase3_multiagent.py
```
Outputs
-------
* `multiagent_outputs/board_session_report.md`  
* `multiagent_outputs/board_session.log`        – structured log file.
"""
from __future__ import annotations

import json
import os
import random
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, TypedDict, Tuple

from rich.logging import RichHandler
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import pathlib

current_dir = pathlib.Path(__file__).parent
env_path = current_dir / ".env"
if not env_path.exists():
    print(f"Warning: .env file not found at {env_path}. Ensure OPENAI_API_KEY is set elsewhere.")
load_dotenv(dotenv_path=env_path)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Project‑wide constants. Adjust as desired."""

    # IO
    cluster_json: Path = Path(
        "Vladimir_Kovtunovskiy/homework2-userboard-simulation/cluster_outputs/clusters_data.json"
    )
    output_dir: Path = Path(
        "Vladimir_Kovtunovskiy/homework2-userboard-simulation/multiagent_outputs"
    )

    # LLM
    model_name: str = "gpt-4.1"
    temperature: float = 0.7

    # Simulation
    persona_count: int = 5
    discussion_rounds: int = 3

    # Determinism
    random_seed: int = 42


CFG = Config()
# random.seed(CFG.random_seed)
CFG.output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Logging – pretty console + file
# -----------------------------------------------------------------------------
log_path = CFG.output_dir / "board_session.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True), logging.FileHandler(log_path, "w", "utf‑8")],
)
logger = logging.getLogger(__name__)
logger.info("Logging initialised → %s", log_path)

# -----------------------------------------------------------------------------
# Utility Dataclasses & State Definition
# -----------------------------------------------------------------------------
@dataclass
class FeatureProposal:
    id: int
    description: str

    def md(self) -> str:
        return f"{self.id}. {self.description}"


@dataclass
class Persona:
    name: str
    background: str
    quote: str
    sentiment: str  # positive / neutral / negative
    pain_points: List[str]
    inspired_by_cluster_id: str | None = None 


    @property
    def system_prompt(self) -> str:
        pain_str = ", ".join(self.pain_points)
        return (
            f"You are {self.name}. {self.background} You feel {self.sentiment} about Spotify. "
            f"Your key pain‑points: {pain_str}. Always speak in first person and be authentic."
        )

    def md(self) -> str:
        pain_str = ", ".join(self.pain_points)
        return (
            f"- **{self.name}** ({self.sentiment}) – {self.background}\n  > *{self.quote}*\n  Pain‑points: {pain_str}"
        )

class AgentState(TypedDict):
    """Defines the state passed between graph nodes."""
    selected_clusters: Dict[str, dict]
    features: List[FeatureProposal]
    personas: List[Persona]
    transcript: str # Markdown transcript
    conversation_history: List[BaseMessage] # Structured message history
    summary: str


# -----------------------------------------------------------------------------
# Generic LLM wrapper (thin, so swapping provider later is trivial)
# -----------------------------------------------------------------------------
LLM = ChatOpenAI(model=CFG.model_name, temperature=CFG.temperature)

def ask_llm(prompt: str) -> str:
    """Synchronous LLM call; central place for retries or logging."""
    logger.debug("LLM prompt: %s", prompt[:200] + ("…" if len(prompt) > 200 else ""))
    reply = LLM.invoke([HumanMessage(content=prompt)]).content.strip()
    return reply

# -----------------------------------------------------------------------------
# 1) Read & validate cluster JSON
# -----------------------------------------------------------------------------

def load_cluster_data(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf‑8") as f:
        clusters = json.load(f)
    if not clusters:
        raise ValueError("Cluster file is empty")
    logger.info("Loaded %d clusters", len(clusters))
    return clusters


def pick_top_clusters(clusters: Dict[str, dict], k: int) -> Dict[str, dict]:
    ranked = sorted(
        clusters.items(),
        key=lambda kv: kv[1]["sentiment_dist"].get("negative", 0),
        reverse=True,
    )
    selected = dict(ranked[:k])
    logger.info("Selected clusters for ideation: %s", list(selected.keys()))
    return selected

# -----------------------------------------------------------------------------
# 2) Feature Ideation
# -----------------------------------------------------------------------------

def ideate_features(selected: Dict[str, dict], n: int = 3) -> List[FeatureProposal]:
    cluster_details = []
    for cid, info in selected.items():
        keywords = ', '.join(info['keywords'])
        neg_sentiment = info['sentiment_dist'].get('negative', 0)
        sample_feedback = info['samples'][0] if info['samples'] else 'N/A'
        cluster_details.append(
            f"- **Cluster {cid}**:\n" 
            f"  - Keywords: {keywords}\n"
            f"  - Negative Sentiment Count: {neg_sentiment}\n"
            f"  - Sample Feedback: '{sample_feedback}...'")

    cluster_str = "\n".join(cluster_details)

    prompt = (
        f"You are a Senior Product Manager at Spotify, specializing in user experience. "
        f"Your task is to propose exactly {n} concrete and realistic product features or UX improvements "
        f"that directly address the pain points highlighted in the following user feedback clusters. "
        f"Focus on actionable solutions that enhance user satisfaction.\n\n"
        f"**User Feedback Clusters:**\n"
        f"{cluster_str}\n\n"
        f"**Instructions:**\n"
        f"1. Generate exactly {n} distinct feature proposals.\n"
        f"2. Each proposal should be a clear, concise imperative statement (e.g., 'Implement a sleep timer', 'Improve playlist organization').\n" 
        f"3. Ensure the features directly relate to the pain points identified in the cluster keywords and sample feedback.\n"
        f"4. Return each proposal on a new line, with no prefixes (like numbers or dashes).\n\n"
        f"**Example Output Format:**\n"
        f"Allow collaborative playlist editing in real-time\n"
        f"Introduce higher fidelity audio options for subscribers\n"
        f"Simplify the podcast discovery interface"
    )
    print(prompt)
    raw = ask_llm(prompt)
    print(raw)
    # Filter out potential empty lines or lines that might just be whitespace
    lines = [l.strip("- *\t ") for l in raw.split('\n') if l.strip() and l.strip() != "Example Output Format:"]
    # Ensure we only take up to 'n' valid proposals
    proposals = [FeatureProposal(id=i + 1, description=desc) for i, desc in enumerate(lines[:n])]
    if len(proposals) < n:
        logger.warning(f"LLM generated fewer than {n} proposals. Got {len(proposals)}.")
    logger.info("Feature ideation complete → %s", [p.description for p in proposals])
    return proposals

# -----------------------------------------------------------------------------
# 3) Persona Generation
# -----------------------------------------------------------------------------

# Updated Persona Generation Function
def generate_personas(clusters: Dict[str, dict], count: int) -> List[Persona]:
    """
    Generates user personas by iteratively querying an LLM for each selected cluster.

    Prioritizes clusters with the most negative average sentiment for selection.

    Args:
        clusters: A dictionary where keys are cluster IDs and values are dicts
                  containing cluster details (keywords, avg_sentiment, samples, etc.).
                  Example value: {'keywords': [...], 'avg_sentiment': -0.5, 'samples': [...]}
        count: The desired number of personas to generate.

    Returns:
        A list of generated Persona objects.

    Raises:
        ValueError: If persona generation fails to produce the required number of valid personas
                    when enough clusters were available.
    """
    if not clusters:
        logger.warning("No clusters provided for persona generation.")
        return []

    # --- 1. Strategic Cluster Selection ---
    cluster_list = list(clusters.items()) # Keep track of keys (cluster IDs)

    # Sort clusters by average sentiment (ascending - most negative first)
    # Handle potential missing 'avg_sentiment' by placing them last
    try:
        cluster_list.sort(key=lambda item: item[1].get('avg_sentiment', float('inf')))
        logger.info("Sorted clusters by average sentiment (most negative first).")
    except Exception as e:
        logger.error(f"Failed to sort clusters by sentiment: {e}. Proceeding with original order.")
        # Fallback or decide how to handle sorting errors

    # Determine how many personas to actually generate based on availability
    num_to_select = min(count, len(cluster_list))
    if num_to_select == 0:
        logger.warning("No clusters available to sample for persona generation.")
        return []
    if num_to_select < count:
        logger.warning(f"Requested {count} personas, but only {num_to_select} clusters are available. Generating {num_to_select}.")

    # Select the top N clusters based on the sorted list
    selected_clusters_with_ids = cluster_list[:num_to_select]
    logger.info(f"Selected {num_to_select} clusters for persona generation (IDs: {[cid for cid, _ in selected_clusters_with_ids]})")


    # --- 2. Iterative Persona Generation (per cluster) ---
    personas: List[Persona] = []
    for cluster_id, cluster_data in selected_clusters_with_ids:
        if len(personas) >= count: # Should not happen with num_to_select logic, but safe check
             break 
             
        logger.info(f"--- Generating Persona for Cluster ID: {cluster_id} ---")

        # Prepare details for the *single* cluster prompt
        keywords_str = ", ".join(cluster_data.get('keywords', []))
        # Use distribution if available, otherwise fallback to average
        sentiment_info = ""
        if 'sentiment_distribution' in cluster_data and isinstance(cluster_data['sentiment_distribution'], dict):
             dist_str = ", ".join(f"{k}: {v}" for k, v in cluster_data['sentiment_distribution'].items())
             sentiment_info = f"Sentiment Distribution=[{dist_str}]"
        elif 'avg_sentiment' in cluster_data:
             sentiment_info = f"Avg Sentiment={cluster_data.get('avg_sentiment', 'N/A'):.2f}"
        else:
             sentiment_info = "Sentiment=N/A"

        sample_feedback = cluster_data.get('samples', ['N/A'])[0] # Use the first sample

        cluster_cue = (
            f"Cluster Details: ID={cluster_id} | Keywords=[{keywords_str}] | "
            f"{sentiment_info} | Sample Feedback=\"{sample_feedback}\""
        )

        # Construct the prompt for a SINGLE persona
        prompt = (
            f"You are an expert persona generator. Your task is to create exactly ONE distinct and detailed Spotify user persona. "
            f"The persona MUST strictly follow this format on a single line: "
            f"Name | Realistic Background | Short Quote | Sentiment Label (positive/neutral/negative) | Key Pain Point 1; Key Pain Point 2; ...\n\n"
            f"Base the persona on the provided cluster details, ensuring the persona's background, quote, sentiment, and pain points reflect the cluster's themes (keywords, sentiment, feedback). "
            f"Pain points should be semicolon-separated. Do NOT include any extra text, explanations, or formatting outside the requested single-line structure. Provide exactly ONE line in the specified format."
            f"\nReference Cluster Details:\n"
            f"{cluster_cue}"
        )
        # Optional: Print prompt for debugging
        # print("--- Single Persona Generation Prompt ---")
        # print(prompt)
        # print("--------------------------------------")

        try:
            raw = ask_llm(prompt)
             # Optional: Print raw output for debugging
            # print(f"--- Raw LLM Output for Cluster {cluster_id} ---")
            # print(raw)
            # print("------------------------------------------")

            # Parse the response (expecting one primary line)
            parsed_persona = None
            raw_lines = [line.strip() for line in raw.strip().split('\n') if line.strip()]

            if not raw_lines:
                logger.warning(f"LLM returned empty output for cluster {cluster_id}.")
                continue # Skip to next cluster

            # Try parsing the first non-empty line, be more tolerant if preamble exists
            target_line = ""
            for line in raw_lines:
                 if '|' in line and len(line.split('|')) == 5: # Look for the characteristic format
                      target_line = line
                      break
            
            if not target_line:
                 logger.warning(f"Could not find line with expected format '|' separator in LLM output for cluster {cluster_id}. Output: '{raw}'")
                 continue

            parts = [p.strip() for p in target_line.split("|")]
            if len(parts) == 5:
                name, background, quote, sentiment, pain_points_str = parts
                
                # Clean up quote and sentiment
                quote = quote.strip("'\" ")
                sentiment = sentiment.lower()
                if sentiment not in ["positive", "neutral", "negative"]:
                    logger.warning(f"Skipping persona for cluster {cluster_id} due to invalid sentiment: '{sentiment}'. Line: '{target_line}'")
                    continue # Skip if sentiment is invalid

                # Split and clean pain points
                pain_points = [p.strip() for p in pain_points_str.split(';') if p.strip()]
                if not pain_points:
                    logger.warning(f"Skipping persona for cluster {cluster_id} due to missing pain points. Line: '{target_line}'")
                    continue # Skip if no pain points provided

                parsed_persona = Persona(
                    name=name,
                    background=background,
                    quote=quote,
                    sentiment=sentiment,
                    pain_points=pain_points,
                    inspired_by_cluster_id=cluster_id # Store the link
                )
                personas.append(parsed_persona)
                logger.info(f"Successfully parsed persona '{name}' based on cluster {cluster_id}.")

            else: # Handles cases where a line has '|' but not the right number of parts
                logger.warning(f"Skipping malformed persona line for cluster {cluster_id}. Expected 5 parts separated by '|', got {len(parts)}. Line: '{target_line}'")

        except Exception as e:
            logger.error(f"Error processing LLM response for cluster {cluster_id}: {e}", exc_info=True)
            # Decide if you want to retry or just continue

    # --- 3. Final Validation and Return ---
    final_persona_count = len(personas)
    logger.info(f"Generated {final_persona_count} personas successfully out of {num_to_select} selected clusters.")

    # Strict check: Raise error if we didn't get the number we *aimed* for (count, adjusted for cluster availability)
    if final_persona_count < num_to_select:
         logger.error(f"Failed to generate personas for all selected clusters. Expected {num_to_select}, got {final_persona_count}. Some LLM calls or parsing likely failed.")
         # Decide whether to raise an error or return the partial list
         # Raising error if we couldn't even fulfill the selection count
         raise ValueError(f"Persona generation failed for some selected clusters. Expected {num_to_select}, got {final_persona_count}.")

    # If we simply had fewer clusters than requested initially, it's not an error state here.
    # The earlier warning covers this.

    return personas

# -----------------------------------------------------------------------------
# 4) Board Simulation (round‑robin)
# -----------------------------------------------------------------------------

def simulate_board(personas: Sequence[Persona], features: Sequence[FeatureProposal], rounds: int) -> Tuple[str, List[BaseMessage]]:
    """
    Simulates a user board discussion with more dynamic facilitation across rounds.

    Args:
        personas: Sequence of Persona objects.
        features: Sequence of FeatureProposal objects.
        rounds: Number of discussion rounds.

    Returns:
        A tuple containing:
        - str: Formatted markdown transcript of the discussion.
        - List[BaseMessage]: The structured conversation history objects.
    """
    if not personas:
        logger.warning("No personas provided for board simulation.")
        return ("", [])
    if not features:
        logger.warning("No features provided for board simulation.")
        return ("", [])

    features_md = "\n".join([f"{i+1}. {f.description}" for i, f in enumerate(features)]) # Assuming FeatureProposal has a 'description' attribute

    # Initialize persona LLM clients (consider reusing clients if possible)
    persona_llms = {
        p.name: ChatOpenAI(model=CFG.model_name, temperature=CFG.temperature)
        for p in personas
    }

    # --- Define Round-Specific Facilitator Prompts ---
    round_prompts = [
        # Round 1: Initial Reactions
        (f"Okay everyone, thanks for joining. We're looking at {len(features)} potential new features for Spotify based on user feedback:\n{features_md}\n\n"
         "Let's go around once. Please give your initial, honest opinion on these features in 1-3 sentences. What's your gut reaction?"),

        # Round 2: Deeper Dive & Concerns
        ("Thanks for those initial thoughts. Now, let's dig a bit deeper. Thinking about these features:\n"
         "*   Are there any potential downsides, challenges, or things that worry you about implementing them?\n"
         "*   How well do you feel they address the core problems you (or users like you) face?\n"
         "Please share your thoughts, building on what you've heard if relevant."),

        # Round 3: Prioritization & Final Thoughts
        ("Great discussion. For our final round, I want to focus on priority.\n"
         "*   If Spotify could only implement ONE of these features soon, which one would be the MOST important for you and why?\n"
         "*   Are there any final comments or trade-offs you'd want Spotify to consider?\n"
         "Focus on making a choice for priority.")
    ]
    # Add more rounds or default prompts if rounds > len(round_prompts)
    if rounds > len(round_prompts):
        default_prompt = "Continuing the discussion, please share any further thoughts or reactions to the recent comments."
        round_prompts.extend([default_prompt] * (rounds - len(round_prompts)))

    # --- Simulation Loop ---
    conversation_history: List[BaseMessage] = []
    transcript_md_parts: List[str] = ["# 💬 Discussion Transcript", "```markdown"] # Start markdown block

    for round_num in range(rounds):
        # Facilitator introduces the round's focus
        facilitator_prompt = round_prompts[round_num]
        transcript_md_parts.append(f"\n### 🎤 [Facilitator - Round {round_num + 1}]")
        transcript_md_parts.append(facilitator_prompt)
        # Add facilitator prompt to the *shared* history as a HumanMessage (representing the user input driving the conversation)
        conversation_history.append(HumanMessage(content=facilitator_prompt))

        # Round-robin through personas for this round
        for persona in personas:
            logger.info(f"Simulating turn for {persona.name} in round {round_num + 1}")
            agent = persona_llms[persona.name]

            # Prepare messages for the persona's LLM call:
            # Their specific system prompt + the *entire shared* conversation history up to this point
            messages_for_llm = [
                SystemMessage(content=persona.system_prompt), # Persona's identity and perspective
            ] + conversation_history # The shared discussion history

            try:
                # Get the persona's response
                response = agent.invoke(messages_for_llm)
                reply_content = response.content.strip()

                # Log and add to transcript/history
                logger.debug(f"Raw reply from {persona.name}: {reply_content}")
                transcript_md_parts.append(f"\n#### 👤 {persona.name}")
                transcript_md_parts.append(reply_content)
                # Add the persona's response as an AIMessage to the shared history
                conversation_history.append(AIMessage(content=reply_content))

            except Exception as e:
                logger.error(f"Error invoking LLM for persona {persona.name}: {e}", exc_info=True)
                # Add an error message to transcript? Or skip persona?
                transcript_md_parts.append(f"\n#### 👤 {persona.name}")
                transcript_md_parts.append(f"_{persona.name} encountered an technical difficulty and could not respond._")
                # Add a placeholder to history? Or just log? For simplicity, just log and move on.
                conversation_history.append(AIMessage(content=f"Error: Could not generate response for {persona.name}."))


    transcript_md_parts.append("```") # End markdown block
    final_transcript_md = "\n".join(transcript_md_parts)

    logger.info(f"Board simulation complete ({len(conversation_history)} total messages)")
    return final_transcript_md, conversation_history

# -----------------------------------------------------------------------------
# 5) Meeting Summary
# -----------------------------------------------------------------------------

def summarise_meeting(transcript_md: str, conversation_history: List[BaseMessage] = None) -> str:
    """
    Summarizes the virtual board meeting transcript.

    Args:
        transcript_md: The formatted markdown transcript.
        conversation_history: Optional structured history (can be used if needed,
                              but transcript_md is usually sufficient for summarization).

    Returns:
        str: Markdown formatted meeting summary.
    """
    # Using transcript_md as it's already formatted.
    # Could potentially use conversation_history for more structured input if the LLM handles it better.
    prompt = (
        "You are an expert meeting summarizer. Analyze the virtual user board meeting transcript provided below.\n"
        "Your summary MUST include these sections in markdown format:\n\n"
        "1.  **Pros & Cons per Feature:** For each proposed feature, list the key advantages (Pros) and disadvantages or concerns (Cons) raised by the participants. Be specific.\n"
        "2.  **Overall Sentiment & Key Takeaways per Persona:** Briefly describe each persona's overall stance, highlighting their main points, priorities, or key concerns.\n"
        "3.  **Points of Agreement & Disagreement:** Note any areas where personas strongly agreed or disagreed with each other.\n"
        "4.  **Final Recommendation:** Provide a concise (1-3 sentences) go/no-go/conditional recommendation for the features, explicitly mentioning the rationale based on the discussion (e.g., priority, concerns raised).\n\n"
        "---\n"
        "# Meeting Transcript\n"
        f"{transcript_md}\n"
        "---\n"
        "Generate the summary now."
    )

    # Replace with your actual LLM call
    summary = ask_llm(prompt)
    logger.info("Generated meeting summary")
    return summary

# -----------------------------------------------------------------------------
# 6) Report Writer
# -----------------------------------------------------------------------------

def write_report(selected_clusters: Dict[str, dict], features: Sequence[FeatureProposal], personas: Sequence[Persona], transcript: str, summary: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report_path = CFG.output_dir / "board_session_report.md"
    
    with report_path.open("w", encoding="utf-8") as f:
        # Title and metadata
        f.write(f"# 🎵 Spotify Virtual User-Board Session\n\n")
        f.write(f"*Generated on {ts}*\n\n")
        f.write("## 📊 Overview\n\n")
        f.write(f"- **Number of Features Discussed**: {len(features)}\n")
        f.write(f"- **Number of Personas**: {len(personas)}\n")
        f.write(f"- **Discussion Rounds**: {CFG.discussion_rounds}\n\n")
        
        # Selected Clusters
        f.write("## 🔍 Selected User Feedback Clusters\n\n")
        for cluster_id, cluster_data in selected_clusters.items():
            f.write(f"### Cluster {cluster_id}\n\n")
            keywords = cluster_data.get('keywords', [])
            f.write(f"- **Keywords**: {', '.join(keywords)}\n")
            sentiment_dist = cluster_data.get('sentiment_dist', {})
            f.write(f"- **Sentiment Distribution**:\n")
            for sentiment, count in sentiment_dist.items():
                f.write(f"  - {sentiment.capitalize()}: {count}\n")
            f.write(f"- **Sample Feedback**:\n")
            samples = cluster_data.get('samples', [])
            for sample in samples[:2]:
                f.write(f"  > {sample}\n")
            f.write("\n")
        
        # Features
        f.write("## 💡 Proposed Features\n\n")
        for feat in features:
            f.write(f"### {feat.md()}\n\n")
        
        # Personas
        f.write("## 👥 User Personas\n\n")
        for persona in personas:
            f.write(f"### {persona.name}\n\n")
            f.write(f"*{persona.quote}*\n\n")
            f.write(f"**Background**: {persona.background}\n\n")
            f.write(f"**Sentiment**: {persona.sentiment.capitalize()}\n\n")
            f.write("**Key Pain Points**:\n")
            for pain in persona.pain_points:
                f.write(f"- {pain}\n")
            f.write("\n")
        
        # Discussion Transcript
        f.write("## 💬 Discussion Transcript\n\n")
        f.write("```markdown\n")
        # Split transcript into sections for better readability
        sections = transcript.split("\n")
        for section in sections:
            if section.startswith("[Facilitator]"):
                f.write(f"\n### 🎤 {section}\n")
            elif section.startswith("["):
                try:
                    persona_name = section.split("]")[0][1:]
                    message = section.split(']', 1)[1].strip()
                    f.write(f"\n#### 👤 {persona_name}\n")
                    f.write(f"{message}\n")
                except IndexError:
                    f.write(f"{section}\n") # Fallback if format is unexpected
            else:
                f.write(f"{section}\n")
        f.write("```\n\n")
        
        # Summary
        f.write("## 📝 Meeting Summary\n\n")
        f.write(summary + "\n")
        
        # Footer
        f.write("\n---\n")
        f.write("*This report was generated using AI-powered user board simulation.*\n")
    
    logger.info("Markdown report written → %s", report_path)

# -----------------------------------------------------------------------------
# LANGGRAPH Orchestration – keeps each stage modular & observable
# -----------------------------------------------------------------------------

def build_pipeline():
    """Builds the LangGraph StateGraph pipeline."""
    graph = StateGraph(AgentState) # Use the typed state

    # Define nodes that update the state
    graph.add_node("ideate", lambda s: {"features": ideate_features(s["selected_clusters"])})
    graph.add_node("generate_personas", lambda s: {"personas": generate_personas(s["selected_clusters"], CFG.persona_count)})
    
    # Update board node to handle tuple output
    def run_board_simulation(state: AgentState) -> Dict[str, any]:
        transcript_md, history = simulate_board(state["personas"], state["features"], CFG.discussion_rounds)
        return {"transcript": transcript_md, "conversation_history": history}
    graph.add_node("board", run_board_simulation)
    
    # Update summary node to use correct args
    def run_summary_generation(state: AgentState) -> Dict[str, str]:
        summary = summarise_meeting(state["transcript"], state["conversation_history"])
        return {"summary": summary}
    graph.add_node("generate_summary", run_summary_generation)

    # Define edges
    graph.set_entry_point("ideate") # Set the entry point
    graph.add_edge("ideate", "generate_personas")
    graph.add_edge("generate_personas", "board")
    graph.add_edge("board", "generate_summary")
    graph.add_edge("generate_summary", END) # End the graph after summary

    # Compile the graph
    return graph.compile()

# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    clusters = load_cluster_data(CFG.cluster_json)
    selected_clusters_data = pick_top_clusters(clusters, k=CFG.persona_count)  

    pipeline = build_pipeline()

    # Define the initial state using the AgentState structure
    initial_state: AgentState = {
        "selected_clusters": selected_clusters_data,
        "features": [],  # Initialize empty lists/strings for fields populated by the graph
        "personas": [],
        "transcript": "",
        "conversation_history": [], # Initialize conversation history
        "summary": ""
    }

    # Invoke the pipeline with the initial state
    final_state = pipeline.invoke(initial_state)

    write_report(
        final_state["selected_clusters"], 
        final_state["features"], 
        final_state["personas"], 
        final_state["transcript"], 
        final_state["summary"]
    )
    logger.info("✅ Multi‑agent pipeline finished. Portfolio artefact ready.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        raise
