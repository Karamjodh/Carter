import asyncio
from dotenv import load_dotenv
load_dotenv()

from app.services.llm import generate_report

async def test():
    prompt = "In two sentences, what is customer segmentation and why do marketers use it?"

    print("Testing Groq...")
    result = await generate_report(prompt, model="groq")
    print(f"Model:         {result['model_used']}")
    print(f"Input tokens:  {result['input_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")
    print(f"Response:\n{result['text']}")

asyncio.run(test())