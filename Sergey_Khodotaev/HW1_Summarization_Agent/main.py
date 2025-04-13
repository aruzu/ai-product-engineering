import asyncio
import pandas as pd
import json
from extractor_agent import create_extractor_agent
from summary_output import SummaryOutput
from agents import Runner
from comparison_judge_agent import create_comparison_judge_agent

async def main():

    df = pd.read_csv("Reviews.csv")
    feedback_list = df["Text"].sample(n=5, random_state=42).tolist()

    for idx, feedback in enumerate(feedback_list, 1):
        print(f"\n===== FEEDBACK #{idx} =====")
        print(f"\nORIGINAL: {feedback}")


        extractor_agent = create_extractor_agent()
        extractor_result = await Runner.run(extractor_agent, feedback)
        assert isinstance(extractor_result.final_output, SummaryOutput)

        summary_output = extractor_result.final_output

        print(f"\nEXTRACTIVE SUMMARY: {summary_output.extractive_summary}")
        print(f"\nABSTRACTIVE SUMMARY: {summary_output.abstractive_summary}")

        json_input = json.dumps(extractor_result.final_output.model_dump(), indent=2)

        judge_agent = create_comparison_judge_agent()       
        judge_result = await Runner.run(judge_agent, json_input)

        print("\n===== Comparison: =====")
        print(f"\nJudge result: {judge_result.final_output}")

# Запуск
if __name__ == "__main__":
    asyncio.run(main())