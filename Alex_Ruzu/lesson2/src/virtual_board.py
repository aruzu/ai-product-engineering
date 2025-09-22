"""
Enhanced Virtual User Board with Structured Interview Rounds
Based on Vladimir's structured approach adapted for cluster-derived personas
"""

import os
import json
import logging
import random
from datetime import datetime
from typing import List, Dict, Any, Sequence, Tuple
from pathlib import Path

from agents import Agent, Runner, ModelSettings
from dotenv import load_dotenv

from data_types import Persona, FeatureProposal
from utils import safe_file_read, safe_file_write, PipelineError, setup_logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class VirtualUserBoard:
    """Enhanced Virtual User Board with structured interview rounds"""

    def __init__(self, output_dir: str = "docs"):
        """Initialize the virtual board with enhanced facilitator"""
        self.output_dir = Path(output_dir)

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Core interview questions adapted from Vladimir's approach
        self.core_questions = [
            "What are your initial thoughts on these features?",
            "What are the pros and cons of each feature?",
            "What's the one feature that would most improve your experience and why?"
        ]

        self.max_followups = 1

        # Create enhanced facilitator agent
        self.facilitator = Agent(
            name="Facilitator",
            instructions="""
You are a professional user research facilitator running a structured discussion about Viber features.
Your role is to:
1. Ask clear, focused questions about the features
2. Keep the discussion on track while allowing natural conversation
3. Ensure each participant gets a chance to speak
4. Ask meaningful follow-up questions when responses are interesting or unclear
5. Reference specific points from participants' responses in follow-ups

Important guidelines:
- Start with a friendly greeting to the group
- Address the group naturally (e.g., "everyone", "folks", "team")
- Keep the tone conversational but professional
- Don't repeat yourself - use different openers and act naturally
- When asking follow-ups, reference specific points from the participant's response
- Only ask follow-ups that would add genuine value to the discussion
""",
            model="gpt-4o-mini"
        )

    def get_random_llm_config(self) -> Dict[str, Any]:
        """Returns random LLM configuration for persona variety"""
        models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]
        temperature = random.uniform(0.3, 1.0)
        model = random.choice(models)

        return {
            "model": model,
            "temperature": temperature
        }

    def generate_persona_agents(self, personas: List[Persona]) -> List[Agent]:
        """Generate agents for each persona with enhanced instructions"""
        agents = []

        # Enhanced system prompt template
        system_prompt_template = """
You are {persona_name}. Act and respond authentically based on this profile:
- Background: {persona_background}
- Profile: {persona_profile}
- Usage Pattern: {persona_usage_pattern}
- Key Pain Points: {persona_pain_points}

You are taking part in a structured panel interview about Viber features with other users.

For every question you receive:
1. Provide your honest answer based on your profile and pain points
2. If you see responses from other panelists that relate to your experience, briefly mention their name and react
3. Explain WHY you think that way - give specific context from your background
4. Be authentic to your character but stay focused on the topic

Important rules:
- Never mention other people unless you've actually seen their responses
- If no other responses are shown, focus only on your own perspective
- Keep responses ≤120 words unless asked for more detail
- Don't repeat yourself - use natural conversation patterns
- Ground all opinions in your specific background and pain points
"""

        for persona in personas:
            # Get random configuration for variety
            llm_config = self.get_random_llm_config()

            agent = Agent(
                name=persona.name,
                model=llm_config["model"],
                model_settings=ModelSettings(temperature=llm_config["temperature"]),
                tools=[],
                instructions=system_prompt_template.format(
                    persona_name=persona.name,
                    persona_background=persona.background,
                    persona_profile=persona.profile,
                    persona_usage_pattern=persona.usage_pattern,
                    persona_pain_points="; ".join(persona.pain_points)
                )
            )
            agents.append(agent)

        return agents

    async def simulate_userboard(
        self,
        personas: List[Persona],
        features: List[FeatureProposal],
        rounds: int = 3
    ) -> Tuple[str, List[str]]:
        """Run structured board simulation with multiple rounds"""

        if not personas or not features:
            logger.warning(f"Missing personas ({len(personas)}) or features ({len(features)})")
            return "", []

        logger.info(f"Initializing structured simulation with {len(personas)} personas and {len(features)} features")

        # Initialize persona agents
        agents = self.generate_persona_agents(personas)

        # Initialize transcript and conversation history
        transcript: List[str] = []
        conversation_history: List[str] = []

        # Feature list for context
        feature_list_md = "\n".join(f"{i+1}. {f.name}: {f.description}" for i, f in enumerate(features))

        # Welcome message
        welcome_prompt = f"""
We're discussing these Viber features today:
{feature_list_md}

Please welcome the panel and start the discussion.
"""

        welcome_response = await Runner.run(self.facilitator, welcome_prompt)
        transcript.append(f"\n### 🎤 Facilitator – Welcome")
        transcript.append(welcome_response.final_output)
        conversation_history.append(welcome_response.final_output)

        # Run structured interview rounds
        for round_num in range(min(rounds, len(self.core_questions))):
            logger.info(f"Starting round {round_num + 1}/{rounds}")

            # Get the core question for this round
            question = self.core_questions[round_num]

            # Facilitator asks the question
            question_prompt = f"""
We are discussing these Viber features:
{feature_list_md}

Please ask this question to the panel:
{question}
"""

            facilitator_response = await Runner.run(self.facilitator, question_prompt)
            transcript.append(f"\n### 🎤 Facilitator – Round {round_num + 1}")
            transcript.append(facilitator_response.final_output)
            conversation_history.append(facilitator_response.final_output)

            # Each persona responds to the question
            round_responses = []
            for agent in agents:
                # Build context from recent conversation
                recent_context = "\n".join(conversation_history[-3:])  # Last 3 messages

                agent_prompt = f"""
Recent discussion:
{recent_context}

As {agent.name}, please respond to the facilitator's question about the Viber features.
"""

                agent_response = await Runner.run(agent, agent_prompt)
                transcript.append(f"\n#### 👤 {agent.name}")
                transcript.append(agent_response.final_output)
                conversation_history.append(f"{agent.name}: {agent_response.final_output}")
                round_responses.append((agent, agent_response.final_output))

                # Facilitator may ask follow-up questions
                followup_count = 0
                while followup_count < self.max_followups:
                    followup_prompt = f"""
Previous discussion context:
{recent_context}

{agent.name}'s response: {agent_response.final_output}

Based on {agent.name}'s response, would you like to ask a relevant follow-up question?
Only ask if it would add genuine value to the discussion.
If you don't want to ask a follow-up, simply respond with "No follow-up needed."
"""

                    followup_response = await Runner.run(self.facilitator, followup_prompt)

                    if "no follow-up" in followup_response.final_output.lower():
                        break

                    transcript.append(f"\n### 🎤 Facilitator – Follow-up")
                    transcript.append(followup_response.final_output)
                    conversation_history.append(followup_response.final_output)

                    # Get agent's response to follow-up
                    followup_context = "\n".join(conversation_history[-2:])
                    agent_followup_prompt = f"""
{followup_context}

As {agent.name}, please respond to the facilitator's follow-up question.
"""

                    agent_followup_response = await Runner.run(agent, agent_followup_prompt)
                    transcript.append(f"\n#### 👤 {agent.name}")
                    transcript.append(agent_followup_response.final_output)
                    conversation_history.append(f"{agent.name}: {agent_followup_response.final_output}")

                    followup_count += 1

        # Closing
        closing_prompt = """
Thank the panel and provide a brief closing for this user feedback session.
"""
        closing_response = await Runner.run(self.facilitator, closing_prompt)
        transcript.append(f"\n### 🎤 Facilitator – Closing")
        transcript.append(closing_response.final_output)
        conversation_history.append(closing_response.final_output)

        final_transcript_md = "\n".join(transcript)
        logger.info(f"Structured simulation completed – {len(conversation_history)} messages total")

        return final_transcript_md, conversation_history

    async def run_board_simulation_from_files(self) -> Dict[str, Any]:
        """Run simulation using generated personas and features from files"""
        logger.info("Starting virtual board simulation from generated files")

        try:
            # Load personas
            personas_file = self.output_dir / "generated_personas.json"
            if not personas_file.exists():
                return {"success": False, "error": f"Personas file not found: {personas_file}"}

            personas_json = safe_file_read(str(personas_file))
            personas_data = json.loads(personas_json)

            # Convert to Persona objects
            personas = []
            for persona_dict in personas_data:
                try:
                    persona = Persona.from_dict(persona_dict)
                    personas.append(persona)
                except Exception as e:
                    logger.warning(f"Skipping invalid persona: {e}")

            if not personas:
                return {"success": False, "error": "No valid personas found"}

            # Load features
            features_file = self.output_dir / "generated_features.json"
            if not features_file.exists():
                return {"success": False, "error": f"Features file not found: {features_file}"}

            features_json = safe_file_read(str(features_file))
            features_data = json.loads(features_json)

            # Convert to FeatureProposal objects
            features = []
            for feature_dict in features_data:
                try:
                    feature = FeatureProposal.from_dict(feature_dict)
                    features.append(feature)
                except Exception as e:
                    logger.warning(f"Skipping invalid feature: {e}")

            if not features:
                return {"success": False, "error": "No valid features found"}

            logger.info(f"Loaded {len(personas)} personas and {len(features)} features")

            # Run the simulation
            transcript, conversation_history = await self.simulate_userboard(personas, features, rounds=3)

            if not transcript:
                return {"success": False, "error": "Simulation produced no transcript"}

            # Generate summary
            summary = await self.generate_meeting_summary(transcript, personas, features)

            # Save results
            await self.save_simulation_results(transcript, summary, personas, features)

            logger.info("Virtual board simulation completed successfully")
            return {
                "success": True,
                "transcript": transcript,
                "summary": summary,
                "personas_count": len(personas),
                "features_count": len(features)
            }

        except Exception as e:
            logger.error(f"Virtual board simulation failed: {e}")
            return {"success": False, "error": str(e)}

    async def generate_meeting_summary(
        self,
        transcript: str,
        personas: List[Persona],
        features: List[FeatureProposal]
    ) -> str:
        """Generate comprehensive meeting summary"""

        summary_prompt = f"""
Analyze this user feedback session transcript and create a comprehensive summary.

TRANSCRIPT:
{transcript}

PARTICIPANTS:
{chr(10).join(f"- {p.name}: {p.profile}" for p in personas)}

FEATURES DISCUSSED:
{chr(10).join(f"- {f.name}: {f.description}" for f in features)}

Create a summary with these sections:

1. **Executive Summary**: Brief overview of the session and key insights

2. **Feature Feedback**: For each feature, summarize:
   - Overall sentiment (positive/negative/mixed)
   - Key benefits mentioned
   - Main concerns or issues raised
   - Specific user quotes that capture sentiment

3. **Persona Insights**: For each participant, summarize:
   - Their overall stance and priorities
   - Most important pain points they expressed
   - How their background influenced their feedback

4. **Cross-Cutting Themes**: Common themes across multiple participants

5. **Recommendations**: Based on the discussion:
   - Which features should be prioritized and why
   - What modifications might address concerns
   - Areas for further research or validation

Format as clear, professional markdown suitable for product stakeholders.
"""

        try:
            # Use facilitator agent for consistent summarization
            summary_response = await Runner.run(self.facilitator, summary_prompt)
            return summary_response.final_output
        except Exception as e:
            logger.error(f"Failed to generate meeting summary: {e}")
            return f"Summary generation failed: {e}"

    async def save_simulation_results(
        self,
        transcript: str,
        summary: str,
        personas: List[Persona],
        features: List[FeatureProposal]
    ) -> None:
        """Save simulation results to files"""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create comprehensive report
        report_content = f"""# 📱 Viber Virtual User Board Session

*Generated on {timestamp}*

## 📊 Overview

- **Number of Features Discussed**: {len(features)}
- **Number of Participants**: {len(personas)}
- **Session Type**: Structured Interview (3 rounds)
- **Method**: Cluster-based personas with authentic user feedback

## 👥 Participants

{chr(10).join(f"### {p.name}{chr(10)}**Profile:** {p.profile}{chr(10)}**Background:** {p.background}{chr(10)}**Key Pain Points:** {', '.join(p.pain_points)}{chr(10)}" for p in personas)}

## 💡 Features Discussed

{chr(10).join(f"### {f.name}{chr(10)}**Description:** {f.description}{chr(10)}**Problem Addressed:** {f.problem_addressed}{chr(10)}**Value Proposition:** {f.value_proposition}{chr(10)}" for f in features)}

## 💬 Session Transcript

```markdown
{transcript}
```

## 📝 Meeting Summary

{summary}

---
*This report was generated using an enhanced AI-powered user board simulation with cluster-derived personas from real user data.*
"""

        # Save comprehensive report
        report_file = self.output_dir / "virtual_user_board_summary.md"
        safe_file_write(str(report_file), report_content)

        logger.info(f"Simulation results saved to {report_file}")