from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import os
import kagglehub
import pandas as pd
from PIL import Image
import base64
import io
import dotenv
from tools import extractive_summarization, abstractive_summarization
from agent import Agent

dotenv.load_dotenv()

key = os.getenv("OPENAI_API_KEY")

# Download latest version
path = kagglehub.dataset_download("arhamrumi/amazon-product-reviews")

print("Path to dataset files:", path)

def save_markdown(report: str, name: str):
    with open(f"{name}.md", "w") as f:
        f.write(report)
        
def process_image(image_path: str) -> str:
    print(f"Processing image: {image_path}")
    image = Image.open(image_path)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"
        
def main():
    dataset = pd.read_csv(os.path.join(path, "reviews.csv"))
    first_two_texts = [text for text in dataset.Text[:2]]
    review_images = [
        "review_1.jpg",
        "review_2.jpg",
    ]
    inputs = review_images + first_two_texts
    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o-mini", api_key=key),
        tools=[extractive_summarization, abstractive_summarization],
        system_message="""
        You are a research assistant analyzing text summarization approaches. 
        You are capable of analyzing text and images.
        You must ALWAYS use tools when summarizing text or image.
        When asked to summarize text or image:
        1. Use both extractive_summarization and abstractive_summarization tools
        2. Compare the results focusing on:
        - Coherence and readability
        - Information retention
        - Length and conciseness
        - Style differences
        3. Present your findings in a structured report format
        """
    )
    for i, text in enumerate(inputs):
        print(f"Processing {i} text, is image: {os.path.exists(text)}")
        messages = [HumanMessage(content=text)]
        if os.path.exists(text):
            base64_image = process_image(text)
            messages = [
                HumanMessage(content=[
                    {
                            "type": "image_url",
                            "image_url": {"url": base64_image},
                        }
                    ]
                )
            ]
        result = agent.graph.invoke({"messages": messages})
        save_markdown(result["report"], f"report_{i}")

if __name__ == "__main__":
    main()