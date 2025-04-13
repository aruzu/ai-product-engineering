from openai import OpenAI
import base64

class AbstractiveSummarizer:
    """
    Abstractive summarization using OpenAI's GPT model.
    This is a probabilistic approach that generates new text as a summary.
    """
    
    def __init__(self, api_key):
        """Initialize with OpenAI API key from environment or parameter."""
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    def summarize(self, text):
        """Generate an abstractive summary using OpenAI's GPT model."""
        try:
            user_message = {"role": "user", "content": f"Please provide a concise summary of the following text: {text}"}
                
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                    user_message
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in abstractive summarization: {e}")
            return "Error generating abstractive summary."
        
    def __encode_image__(self, image_path):
        """Encode an image to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

