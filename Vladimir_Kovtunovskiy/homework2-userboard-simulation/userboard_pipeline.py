"""
Multi-Agent User Board Simulation Pipeline

This script orchestrates an end-to-end pipeline:
1. Loads clustered user review data.
2. Selects top clusters based on negative sentiment.
3. Uses an LLM to ideate features addressing pains in selected clusters.
4. Generates distinct user personas based on selected clusters using an LLM.
5. Simulates a multi-round user board discussion using LLM agents representing personas.
6. Summarizes the discussion using an LLM.
7. Writes a final Markdown report.

Setup:
1. Ensure Python 3.9+ is installed.
2. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt` (Create this file!)
4. Create a `.env` file in the same directory as the script with your `OPENAI_API_KEY="sk-..."`.
5. Place the cluster JSON file (e.g., `clusters_data.json`) in a `cluster_outputs` subdirectory relative to this script.
6. Run the script: `python your_script_name.py`

Outputs:
* `multiagent_outputs/board_session_report.md` - The final analysis report.
* `multiagent_outputs/board_session.log` - Detailed execution log.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, TypedDict

import numpy as np # Added for seeding
from dotenv import load_dotenv
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback # Optional: for cost tracking
from langchain_core.exceptions import LangChainException # For broader error catching
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from rich.logging import RichHandler
# Added for retry logic
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import APIError # Specific error type

# --- Determine Project Root and Load Env ---
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR

env_path = PROJECT_ROOT / ".env"
if not env_path.exists():
    print(f"Warning: .env file not found at {env_path}. Ensure OPENAI_API_KEY is set via environment variables.")
load_dotenv(dotenv_path=env_path)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Project-wide constants. Adjust as desired."""

    # IO - Paths relative to the PROJECT_ROOT defined above
    cluster_json: Path = PROJECT_ROOT / "cluster_outputs" / "clusters_data.json"
    output_dir: Path = PROJECT_ROOT / "multiagent_outputs"

    # LLM - Use a valid OpenAI model name
    model_name: str = "gpt-4-turbo" # Example: Use a valid model
    temperature: float = 0.7

    # Simulation
    persona_count: int = 5
    feature_count: int = 3 # Define number of features to ideate
    discussion_rounds: int = 3

    # Determinism - Set to None to disable seeding
    random_seed: int | None = 42


CFG = Config()
CFG.output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Logging – pretty console + file
# -----------------------------------------------------------------------------
log_path = CFG.output_dir / "board_session.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s", # Added module/lineno
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False), # show_path=False for cleaner console
              logging.FileHandler(log_path, "w", "utf-8")],
    force=True # Override root logger config if necessary
)
# Reduce verbosity of noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__) # Get logger for this module
logger.info("Logging initialised. Log file: %s", log_path)
logger.info("Using configuration: %s", CFG)


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
        """Generates the system prompt for the LLM agent representing this persona."""
        pain_str = "; ".join(self.pain_points) # Use semicolon for clarity if needed
        return (
            f"You are {self.name}. Act and respond authentically based on this profile:\n"
            f"- Background: {self.background}\n"
            f"- Overall Sentiment towards Spotify: {self.sentiment}\n"
            f"- Key Pain Points/Frustrations: {pain_str}\n"
            f"Always speak in the first person ('I', 'me', 'my'). Keep your responses concise (1-3 sentences unless asked otherwise) and focused on the discussion topic. "
            f"Ground your opinions in your background and pain points. Be honest and natural."
        )

    def md(self) -> str:
        """Returns a Markdown representation of the persona."""
        pain_str = "\n  - ".join(self.pain_points)
        return (
            f"### {self.name}\n\n"
            f"*{self.quote}*\n\n"
            f"**Background**: {self.background}\n\n"
            f"**Sentiment**: {self.sentiment.capitalize()}\n\n"
            f"**Key Pain Points**:\n  - {pain_str}\n"
        )


class AgentState(TypedDict):
    """Defines the state passed between graph nodes."""
    selected_clusters: Dict[str, dict]
    features: List[FeatureProposal]
    personas: List[Persona]
    transcript: str # Markdown transcript
    conversation_history: List[BaseMessage] # Structured message history
    summary: str
    error: str | None # To capture errors in the graph


# -----------------------------------------------------------------------------
# Generic LLM wrapper with Retry Logic
# -----------------------------------------------------------------------------
LLM = ChatOpenAI(model=CFG.model_name, temperature=CFG.temperature)

# Implement retry logic for LLM calls
@retry(
    stop=stop_after_attempt(3), # Retry up to 3 times
    wait=wait_fixed(2),         # Wait 2 seconds between retries
    reraise=True                # Re-raise the exception if all retries fail
)
def invoke_llm_with_retry(llm_client: ChatOpenAI, messages: List[BaseMessage]) -> str:
    """Invokes the LLM with retry logic for transient errors."""
    logger.debug("Invoking LLM with %d messages...", len(messages))
    try:
        # Optional: Track token usage and cost
        # with get_openai_callback() as cb:
        response = llm_client.invoke(messages)
            # logger.debug("LLM call stats: %s", cb)
        reply = response.content.strip()
        logger.debug("LLM reply received (first 100 chars): %s", reply[:100])
        return reply
    except APIError as e:
        logger.error("OpenAI API Error during LLM call: %s", e, exc_info=True)
        raise # Let retry handle it or fail
    except LangChainException as e:
        logger.error("LangChain Error during LLM call: %s", e, exc_info=True)
        raise # Let retry handle it or fail
    except Exception as e:
        logger.error("Unexpected error during LLM call: %s", e, exc_info=True)
        raise # Let retry handle it or fail

def ask_llm(prompt: str) -> str:
    """Synchronous LLM call using the retry wrapper."""
    logger.debug("ask_llm prompt (first 200 chars): %s", prompt[:200] + ("…" if len(prompt) > 200 else ""))
    # Use the global LLM client
    return invoke_llm_with_retry(LLM, [HumanMessage(content=prompt)])

# -----------------------------------------------------------------------------
# 1) Read & validate cluster JSON
# -----------------------------------------------------------------------------

def load_cluster_data(path: Path) -> Dict[str, dict]:
    """Loads cluster data from JSON, handling list or dict format."""
    logger.info("Loading cluster data from: %s", path)
    if not path.exists():
        logger.error("Cluster JSON file not found at %s", path)
        raise FileNotFoundError(f"Cluster JSON file not found: {path}")

    try:
        with path.open(encoding="utf-8") as f:
            loaded_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON from %s: %s", path, e)
        raise ValueError(f"Invalid JSON file: {path}") from e
    except Exception as e:
        logger.error("Failed to read cluster file %s: %s", path, e)
        raise IOError(f"Could not read cluster file: {path}") from e

    clusters: Dict[str, dict]
    if isinstance(loaded_data, list):
        # Convert list to dict using string indices as keys.
        # This ensures downstream code using keys (like persona gen) works consistently.
        clusters = {str(i): item for i, item in enumerate(loaded_data)}
        logger.info("Loaded cluster data as list, converted to dict using string indices (0 to %d).", len(clusters) - 1)
    elif isinstance(loaded_data, dict):
        clusters = loaded_data
        logger.info("Loaded cluster data as dict.")
    else:
        logger.error("Unexpected data type loaded from JSON: %s. Expected dict or list.", type(loaded_data))
        raise TypeError(f"Unexpected data type in JSON file: {type(loaded_data)}")

    if not clusters:
        logger.error("Cluster data is empty after loading/processing.")
        raise ValueError("Cluster data is empty.")

    # Basic validation (optional but recommended)
    for key, value in clusters.items():
        if not isinstance(value, dict) or 'keywords' not in value or 'sentiment_dist' not in value:
            logger.warning("Cluster '%s' might be missing expected keys ('keywords', 'sentiment_dist').", key)

    logger.info("Successfully loaded and validated %d clusters.", len(clusters))
    return clusters


def pick_top_clusters(clusters: Dict[str, dict], k: int) -> Dict[str, dict]:
    """Selects top k clusters, ranked by negative sentiment count."""
    if k <= 0:
        return {}
    try:
        # Ensure robustness if sentiment_dist or negative key is missing
        ranked = sorted(
            clusters.items(),
            key=lambda kv: kv[1].get('sentiment_dist', {}).get("negative", 0) if isinstance(kv[1], dict) else 0,
            reverse=True,
        )
    except Exception as e:
        logger.error("Error sorting clusters: %s. Check cluster data format.", e, exc_info=True)
        raise TypeError("Cluster data format seems incompatible for sorting.") from e

    num_to_select = min(k, len(ranked))
    if num_to_select < k:
        logger.warning("Requested %d clusters, but only %d available after ranking.", k, num_to_select)

    selected = dict(ranked[:num_to_select])
    logger.info("Selected top %d clusters for ideation: %s", len(selected), list(selected.keys()))
    return selected

# -----------------------------------------------------------------------------
# 2) Feature Ideation
# -----------------------------------------------------------------------------

def ideate_features(selected: Dict[str, dict], n: int = CFG.feature_count) -> List[FeatureProposal]:
    """Generates feature proposals based on selected clusters using an LLM."""
    logger.info("Starting feature ideation for %d features...", n)
    if not selected:
        logger.warning("No clusters selected for feature ideation.")
        return []

    cluster_details = []
    for cid, info in selected.items():
        # Basic validation of cluster info structure
        if not isinstance(info, dict):
            logger.warning("Skipping cluster '%s' in ideation prompt due to unexpected format: %s", cid, type(info))
            continue

        keywords = ', '.join(info.get('keywords', ['N/A']))
        neg_sentiment = info.get('sentiment_dist', {}).get('negative', 0)
        samples = info.get('samples', [])
        sample_feedback = samples[0] if samples else 'N/A'

        cluster_details.append(
            f"- **Cluster {cid}**:\n"
            f"  - Keywords: {keywords}\n"
            f"  - Negative Sentiment Count: {neg_sentiment}\n"
            f"  - Sample Feedback: '{sample_feedback[:200]}...'") # Limit sample length

    if not cluster_details:
        logger.error("No valid cluster details could be extracted for the feature ideation prompt.")
        return []

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
        f"4. Return EACH proposal on a new line, with NO prefixes (like numbers or dashes) and NO blank lines between proposals.\n\n"
        f"**Example Output Format:**\n"
        f"Allow collaborative playlist editing in real-time\n"
        f"Introduce higher fidelity audio options for subscribers\n"
        f"Simplify the podcast discovery interface"
    )
    # REMOVED print(prompt) - Rely on debug logging in ask_llm

    try:
        raw = ask_llm(prompt)
        # REMOVED print(raw) - Rely on debug logging in ask_llm

        # Process the raw response
        lines = [line.strip() for line in raw.split('\n') if line.strip()]
        # Filter out any potential introductory text or examples
        proposals_text = [line for line in lines if not line.startswith(("*", "-", "#")) and "|" not in line] # Basic filtering

        # Ensure we only take up to 'n' valid proposals
        proposals = [FeatureProposal(id=i + 1, description=desc) for i, desc in enumerate(proposals_text[:n])]

        if len(proposals) < n:
            logger.warning("LLM generated fewer than %d valid proposals. Got %d. Raw output:\n%s", n, len(proposals), raw)
        elif len(proposals) > n:
             logger.warning("LLM generated more than %d proposals. Taking the first %d. Raw output:\n%s", n, n, raw)
             proposals = proposals[:n] # Trim excess

        logger.info("Feature ideation complete. Generated %d features: %s", len(proposals), [p.description for p in proposals])
        return proposals

    except Exception as e:
        logger.error("Feature ideation failed: %s", e, exc_info=True)
        # Return empty list or re-raise depending on desired pipeline behavior
        return []


# -----------------------------------------------------------------------------
# 3) Persona Generation
# -----------------------------------------------------------------------------

# Using the robust iterative persona generation function provided previously
def generate_personas(clusters: Dict[str, dict], count: int) -> List[Persona]:
    """
    Generates user personas by iteratively querying an LLM for each selected cluster.
    Prioritizes clusters based on negative sentiment or other criteria if needed.
    """
    if not clusters:
        logger.warning("No clusters provided for persona generation.")
        return []

    # --- 1. Strategic Cluster Selection (using negative sentiment count) ---
    cluster_list = list(clusters.items())
    try:
        # Sort by negative sentiment count (descending)
        cluster_list.sort(
            key=lambda item: item[1].get('sentiment_dist', {}).get('negative', 0) if isinstance(item[1], dict) else 0,
            reverse=True
        )
        logger.info("Sorted clusters for persona generation based on negative sentiment count.")
    except Exception as e:
        logger.error("Failed to sort clusters for persona generation: %s. Proceeding with original order.", e)

    num_to_select = min(count, len(cluster_list))
    if num_to_select == 0:
        logger.warning("No clusters available to sample for persona generation.")
        return []
    if num_to_select < count:
        logger.warning("Requested %d personas, but only %d clusters available/selected. Generating %d.", count, num_to_select, num_to_select)

    selected_clusters_with_ids = cluster_list[:num_to_select]
    logger.info("Selected %d clusters for persona generation (IDs: %s)", num_to_select, [cid for cid, _ in selected_clusters_with_ids])

    # --- 2. Iterative Persona Generation (per cluster) ---
    personas: List[Persona] = []
    for cluster_id, cluster_data in selected_clusters_with_ids:
        # Defensive check against reaching count limit (shouldn't happen with logic above)
        if len(personas) >= count:
            logger.warning("Reached persona count limit unexpectedly. Stopping generation.")
            break

        logger.info("--- Generating Persona for Cluster ID: %s ---", cluster_id)

        # Validate cluster_data structure
        if not isinstance(cluster_data, dict):
            logger.warning("Skipping cluster '%s' due to invalid data format: %s", cluster_id, type(cluster_data))
            continue

        # Prepare cluster details for the prompt
        keywords_str = ", ".join(cluster_data.get('keywords', ['N/A']))
        sentiment_dist = cluster_data.get('sentiment_dist', {})
        avg_sentiment = cluster_data.get('avg_sentiment') # May be None
        samples = cluster_data.get('samples', [])
        sample_feedback = samples[0] if samples else 'N/A'

        sentiment_info = f"Sentiment Distribution=[{', '.join(f'{k}: {v}' for k, v in sentiment_dist.items())}]"
        if avg_sentiment is not None:
             sentiment_info += f" | Avg Sentiment={avg_sentiment:.2f}"

        cluster_cue = (
            f"Cluster Details: ID={cluster_id} | Keywords=[{keywords_str}] | "
            f"{sentiment_info} | Sample Feedback=\"{sample_feedback[:200]}...\"" # Limit length
        )

        prompt = (
            f"You are an expert persona generator. Your task is to create exactly ONE distinct and detailed Spotify user persona. "
            f"The persona MUST strictly follow this format on a single line (no extra text before or after): "
            f"Name | Realistic Background | Short Quote | Sentiment Label (positive/neutral/negative) | Key Pain Point 1; Key Pain Point 2; ...\n\n"
            f"Base the persona EXCLUSIVELY on the provided cluster details. Ensure the persona's background, quote, sentiment label, and pain points directly reflect the cluster's themes (keywords, sentiment distribution, average sentiment, sample feedback). "
            f"Use the provided 'Sentiment Label' options strictly. Pain points MUST be semicolon-separated.\n\n"
            f"Reference Cluster Details:\n"
            f"{cluster_cue}"
        )

        try:
            raw = ask_llm(prompt)
            raw_lines = [line.strip() for line in raw.strip().split('\n') if line.strip()]

            if not raw_lines:
                logger.warning("LLM returned empty output for cluster %s.", cluster_id)
                continue

            # Find the first line that looks like the persona format
            target_line = ""
            for line in raw_lines:
                 parts = line.split('|')
                 if len(parts) == 5:
                      target_line = line
                      logger.debug("Found potential persona line: %s", target_line)
                      break # Use the first valid line found

            if not target_line:
                 logger.warning("Could not find line with expected format ('|' separator, 5 parts) in LLM output for cluster %s. Output: '%s'", cluster_id, raw)
                 continue

            # Parse the target line
            parts = [p.strip() for p in target_line.split("|")]
            name, background, quote, sentiment, pain_points_str = parts

            # Clean up and validate
            quote = quote.strip("'\" ")
            sentiment = sentiment.lower()
            if sentiment not in ["positive", "neutral", "negative"]:
                logger.warning("Skipping persona for cluster %s due to invalid sentiment: '%s'. Line: '%s'", cluster_id, sentiment, target_line)
                continue

            pain_points = [p.strip() for p in pain_points_str.split(';') if p.strip()]
            if not pain_points:
                logger.warning("Skipping persona for cluster %s due to missing pain points. Line: '%s'", cluster_id, target_line)
                continue

            personas.append(Persona(
                name=name,
                background=background,
                quote=quote,
                sentiment=sentiment,
                pain_points=pain_points,
                inspired_by_cluster_id=cluster_id # Store the link
            ))
            logger.info("Successfully parsed persona '%s' based on cluster %s.", name, cluster_id)

        except Exception as e:
            logger.error("Error processing LLM response or parsing persona for cluster %s: %s", cluster_id, e, exc_info=True)
            # Continue to next cluster on error

    # --- 3. Final Validation and Return ---
    final_persona_count = len(personas)
    logger.info("Generated %d personas successfully out of %d selected clusters.", final_persona_count, num_to_select)

    # Check if we got at least one persona if we expected some
    if num_to_select > 0 and final_persona_count == 0:
        logger.error("Failed to generate any valid personas despite selecting clusters.")
        raise ValueError("Persona generation failed completely.")
    # Check if we got fewer than intended due to errors
    elif final_persona_count < num_to_select:
         logger.warning("Failed to generate personas for all selected clusters. Expected %d, got %d. Some LLM calls or parsing likely failed.", num_to_select, final_persona_count)
         # Decide whether to raise or proceed with fewer personas. Proceeding for robustness.

    return personas

# -----------------------------------------------------------------------------
# 4) Board Simulation (Improved with Dynamic Facilitation & LLM Reuse)
# -----------------------------------------------------------------------------

def simulate_board(personas: Sequence[Persona], features: Sequence[FeatureProposal], rounds: int) -> Tuple[str, List[BaseMessage]]:
    """
    Simulates a user board discussion with dynamic facilitation across rounds,
    reusing a single LLM client.
    """
    if not personas:
        logger.warning("No personas provided for board simulation.")
        return ("", [])
    if not features:
        logger.warning("No features provided for board simulation.")
        return ("", [])

    logger.info("Starting board simulation with %d personas, %d features, %d rounds.", len(personas), len(features), rounds)

    features_md = "\n".join([f"{i+1}. {f.description}" for i, f in enumerate(features)])

    # REUSE the global LLM client for all agents
    agent_llm = LLM

    # --- Define Round-Specific Facilitator Prompts ---
    round_prompts = [
        # Round 1: Initial Reactions
        (f"Okay everyone, thanks for joining. We're looking at {len(features)} potential new features for Spotify based on user feedback:\n{features_md}\n\n"
         "Let's go around once. Please give your initial, honest opinion on these features in 1-3 sentences, grounding it in your own experience. What's your gut reaction?"),

        # Round 2: Deeper Dive & Concerns
        ("Thanks for those initial thoughts. Now, let's dig a bit deeper into these features:\n"
         f"{features_md}\n\n" # Re-iterate features for context
         "*   Are there any potential downsides, implementation challenges, or things that worry you about them?\n"
         "*   How well do you feel they address the core problems you (or users like you) face?\n"
         "Please share your thoughts (1-3 sentences), building on what you've heard if relevant."),

        # Round 3: Prioritization & Final Thoughts
        ("Great discussion. For our final round, let's focus on priority.\n"
         f"{features_md}\n\n" # Re-iterate features
         "*   If Spotify could only implement ONE of these {len(features)} features soon, which one would be the MOST important for you and why?\n"
         "*   Any final comments or trade-offs Spotify should consider?\n"
         "Focus on making a choice for priority (1-3 sentences).")
    ]
    # Ensure enough prompts exist for the number of rounds
    if rounds > len(round_prompts):
        default_prompt = "Continuing the discussion, please share any further thoughts or reactions to the recent comments (1-3 sentences)."
        round_prompts.extend([default_prompt] * (rounds - len(round_prompts)))

    # --- Simulation Loop ---
    conversation_history: List[BaseMessage] = []
    # Use f-string for cleaner markdown block start/end
    transcript_md_parts: List[str] = ["# 💬 Discussion Transcript", "```markdown"]

    for round_num in range(rounds):
        round_idx = round_num + 1
        logger.info("--- Starting Discussion Round %d ---", round_idx)
        facilitator_prompt = round_prompts[round_num]

        # Add facilitator prompt to transcript and history
        transcript_md_parts.append(f"\n### 🎤 [Facilitator - Round {round_idx}]")
        transcript_md_parts.append(facilitator_prompt)
        conversation_history.append(HumanMessage(content=facilitator_prompt))

        # Round-robin through personas
        for persona in personas:
            logger.info("Simulating turn for %s in round %d", persona.name, round_idx)

            # Prepare messages: Persona's system prompt + entire shared history
            messages_for_llm: List[BaseMessage] = [
                SystemMessage(content=persona.system_prompt),
            ] + conversation_history

            try:
                # Use the single shared LLM client with retry logic
                reply_content = invoke_llm_with_retry(agent_llm, messages_for_llm)

                # Add response to transcript and history
                transcript_md_parts.append(f"\n#### 👤 {persona.name}")
                transcript_md_parts.append(reply_content)
                conversation_history.append(AIMessage(content=reply_content))

            except Exception as e:
                # Log error and add placeholder to transcript
                logger.error("Error invoking LLM for persona %s in round %d: %s", persona.name, round_idx, e, exc_info=True)
                error_msg = f"_{persona.name} encountered a technical difficulty and could not respond._"
                transcript_md_parts.append(f"\n#### 👤 {persona.name}")
                transcript_md_parts.append(error_msg)
                # Add placeholder to history to maintain turn structure if needed, or skip
                conversation_history.append(AIMessage(content=f"Error: Could not generate response for {persona.name}."))

    transcript_md_parts.append("```") # End markdown block
    final_transcript_md = "\n".join(transcript_md_parts)

    logger.info("Board simulation complete. Total messages exchanged: %d", len(conversation_history))
    return final_transcript_md, conversation_history

# -----------------------------------------------------------------------------
# 5) Meeting Summary
# -----------------------------------------------------------------------------

def summarise_meeting(transcript_md: str, conversation_history: List[BaseMessage] | None = None) -> str:
    """Summarizes the virtual board meeting transcript using an LLM."""
    logger.info("Generating meeting summary...")
    if not transcript_md.strip():
        logger.warning("Transcript is empty, cannot generate summary.")
        return "Error: Transcript was empty."

    # Use the detailed prompt provided previously
    prompt = (
        "You are an expert meeting summarizer. Analyze the virtual user board meeting transcript provided below.\n"
        "Your summary MUST include these sections in markdown format:\n\n"
        "1.  **Pros & Cons per Feature:** For each proposed feature, list the key advantages (Pros) and disadvantages or concerns (Cons) raised by the participants. Be specific.\n"
        "2.  **Overall Sentiment & Key Takeaways per Persona:** Briefly describe each persona's overall stance, highlighting their main points, priorities, or key concerns.\n"
        "3.  **Points of Agreement & Disagreement:** Note any areas where personas strongly agreed or disagreed with each other.\n"
        "4.  **Final Recommendation:** Provide a concise (1-3 sentences) go/no-go/conditional recommendation for the features, explicitly mentioning the rationale based on the discussion (e.g., priority, concerns raised).\n\n"
        "---\n"
        f"{transcript_md}\n" # Embed the full transcript directly
        "---\n"
        "Generate the summary now."
    )

    try:
        summary = ask_llm(prompt)
        logger.info("Meeting summary generated successfully.")
        return summary
    except Exception as e:
        logger.error("Failed to generate meeting summary: %s", e, exc_info=True)
        return f"Error: Failed to generate meeting summary due to: {e}"


# -----------------------------------------------------------------------------
# 6) Report Writer (Simplified Transcript Handling)
# -----------------------------------------------------------------------------

def write_report(
    selected_clusters: Dict[str, dict],
    features: Sequence[FeatureProposal],
    personas: Sequence[Persona],
    transcript: str, # Expect pre-formatted Markdown transcript
    summary: str
):
    """Writes the final Markdown report to the output directory."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report_path = CFG.output_dir / "board_session_report.md"
    logger.info("Writing final report to: %s", report_path)

    try:
        with report_path.open("w", encoding="utf-8") as f:
            # --- Header ---
            f.write("# 🎵 Spotify Virtual User-Board Session\n\n")
            f.write(f"*Generated on {ts}*\n\n")
            f.write("## 📊 Overview\n\n")
            f.write(f"- **Number of Features Discussed**: {len(features)}\n")
            f.write(f"- **Number of Personas**: {len(personas)}\n")
            f.write(f"- **Discussion Rounds**: {CFG.discussion_rounds}\n\n")

            # --- Selected Clusters ---
            f.write("## 🔍 Selected User Feedback Clusters\n\n")
            # Sort clusters by ID for consistent report order
            for cluster_id, cluster_data in sorted(selected_clusters.items(), key=lambda item: str(item[0])):
                f.write(f"### Cluster {cluster_id}\n\n")
                keywords = cluster_data.get('keywords', [])
                f.write(f"- **Keywords**: {', '.join(keywords)}\n")
                sentiment_dist = cluster_data.get('sentiment_dist', {})
                f.write("- **Sentiment Distribution**:\n")
                # Sort sentiment labels for consistent order
                for sentiment, count in sorted(sentiment_dist.items()):
                    f.write(f"  - {sentiment.capitalize()}: {count}\n")
                f.write("- **Sample Feedback**:\n")
                samples = cluster_data.get('samples', [])
                for sample in samples[:2]: # Show top 2 samples
                    # Clean sample for display (remove extra whitespace)
                    cleaned_sample = re.sub(r'\s+', ' ', str(sample)).strip()
                    f.write(f"  > {cleaned_sample}\n")
                f.write("\n")

            # --- Proposed Features ---
            f.write("## 💡 Proposed Features\n\n")
            if features:
                for feat in features:
                    f.write(f"### {feat.md()}\n\n")
            else:
                f.write("*No features were generated.*\n\n")

            # --- User Personas ---
            f.write("## 👥 User Personas\n\n")
            if personas:
                for persona in personas:
                    # Use the persona's built-in Markdown method
                    f.write(persona.md())
                    f.write("\n") # Add separator
            else:
                f.write("*No personas were generated.*\n\n")

            # --- Discussion Transcript (Write pre-formatted string directly) ---
            f.write("## 💬 Discussion Transcript\n\n")
            if transcript.strip():
                 # No need to parse, just write the markdown string generated by simulate_board
                 f.write(transcript + "\n\n")
            else:
                 f.write("*No discussion transcript was generated.*\n\n")


            # --- Meeting Summary ---
            f.write("## 📝 Meeting Summary\n\n")
            if summary.strip() and not summary.startswith("Error:"):
                 f.write(summary + "\n")
            else:
                 f.write(f"*Summary could not be generated. Details: {summary}*\n")


            # --- Footer ---
            f.write("\n---\n")
            f.write("*This report was generated using an AI-powered user board simulation pipeline.*\n")

        logger.info("Markdown report written successfully.")

    except IOError as e:
        logger.error("Failed to write report file %s: %s", report_path, e, exc_info=True)
    except Exception as e:
        logger.error("An unexpected error occurred during report writing: %s", e, exc_info=True)


# -----------------------------------------------------------------------------
# LANGGRAPH Orchestration – keeps each stage modular & observable
# -----------------------------------------------------------------------------

def build_pipeline():
    """Builds the LangGraph StateGraph pipeline."""
    logger.info("Building LangGraph pipeline...")
    graph = StateGraph(AgentState) # Use the typed state

    # Define node functions that update the state
    def run_feature_ideation(state: AgentState) -> Dict[str, Any]:
        try:
            features = ideate_features(state["selected_clusters"], CFG.feature_count)
            return {"features": features, "error": None}
        except Exception as e:
            logger.error("Error in feature ideation node: %s", e, exc_info=True)
            return {"error": f"Feature Ideation Failed: {e}"}

    def run_persona_generation(state: AgentState) -> Dict[str, Any]:
        if state.get("error"): return {} # Skip if previous step failed
        try:
            personas = generate_personas(state["selected_clusters"], CFG.persona_count)
            return {"personas": personas, "error": None}
        except Exception as e:
            logger.error("Error in persona generation node: %s", e, exc_info=True)
            return {"error": f"Persona Generation Failed: {e}"}

    def run_board_simulation(state: AgentState) -> Dict[str, Any]:
        if state.get("error"): return {} # Skip if previous step failed
        try:
            transcript_md, history = simulate_board(state["personas"], state["features"], CFG.discussion_rounds)
            return {"transcript": transcript_md, "conversation_history": history, "error": None}
        except Exception as e:
            logger.error("Error in board simulation node: %s", e, exc_info=True)
            return {"error": f"Board Simulation Failed: {e}"}

    def run_summary_generation(state: AgentState) -> Dict[str, Any]:
        if state.get("error"): return {} # Skip if previous step failed
        try:
            summary = summarise_meeting(state["transcript"], state["conversation_history"])
            return {"summary": summary, "error": None}
        except Exception as e:
            logger.error("Error in summary generation node: %s", e, exc_info=True)
            return {"error": f"Summary Generation Failed: {e}"}

    # Add nodes to the graph
    graph.add_node("ideate", run_feature_ideation)
    graph.add_node("generate_personas", run_persona_generation)
    graph.add_node("board", run_board_simulation)
    graph.add_node("generate_summary", run_summary_generation)

    # Define edges - standard sequential flow
    graph.set_entry_point("ideate")
    graph.add_edge("ideate", "generate_personas")
    graph.add_edge("generate_personas", "board")
    graph.add_edge("board", "generate_summary")
    graph.add_edge("generate_summary", END)

    # Compile the graph
    pipeline = graph.compile()
    logger.info("LangGraph pipeline compiled successfully.")
    return pipeline

# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    """Main function to load data, build and run the pipeline, and write the report."""
    # Apply random seed early if specified
    if CFG.random_seed is not None:
         random.seed(CFG.random_seed)
         np.random.seed(CFG.random_seed) # Seed numpy as well
         # torch seeding can be added here if needed, see notes above
         logger.info("Using random seed: %d", CFG.random_seed)

    try:
        # --- Data Loading and Selection ---
        clusters = load_cluster_data(CFG.cluster_json)
        # Select clusters based on negative sentiment count for feature/persona generation
        selected_clusters_data = pick_top_clusters(clusters, k=CFG.persona_count)
        if not selected_clusters_data:
            logger.error("No clusters were selected. Cannot proceed.")
            sys.exit(1) # Exit if no clusters selected

        # --- Build and Run Pipeline ---
        pipeline = build_pipeline()

        # Define the initial state
        initial_state: AgentState = {
            "selected_clusters": selected_clusters_data,
            "features": [],
            "personas": [],
            "transcript": "",
            "conversation_history": [],
            "summary": "",
            "error": None # Initialize error state
        }

        logger.info("Invoking pipeline...")
        final_state = pipeline.invoke(initial_state)
        logger.info("Pipeline invocation complete.")

        # --- Error Check ---
        if final_state.get("error"):
            logger.error("Pipeline execution failed with error: %s", final_state["error"])
            # Still attempt to write a partial report if possible
            logger.warning("Attempting to write partial report despite pipeline error...")

        # --- Write Report ---
        # Ensure keys exist even if steps failed
        write_report(
            final_state.get("selected_clusters", selected_clusters_data), # Use initial if final missing
            final_state.get("features", []),
            final_state.get("personas", []),
            final_state.get("transcript", "*Transcript generation failed or skipped due to error.*"),
            final_state.get("summary", f"*Summary generation failed or skipped. Error: {final_state.get('error')}*")
        )

        if final_state.get("error"):
            logger.error("Pipeline finished with errors.")
            sys.exit(1) # Exit with error code
        else:
            logger.info("✅ Multi-agent pipeline finished successfully.")

    except FileNotFoundError as e:
        logger.error("❌ Pipeline failed: Required file not found.")
        logger.error(e, exc_info=True)
        sys.exit(2)
    except ValueError as e:
        logger.error("❌ Pipeline failed: Data validation or processing error.")
        logger.error(e, exc_info=True)
        sys.exit(3)
    except Exception as e:
        logger.error("❌ Pipeline failed with an unexpected error.")
        logger.error(e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()