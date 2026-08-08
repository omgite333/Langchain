from langchain_openai import ChaOpenAI, ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

from Structured_output.buildin.with_structured_output_pydantic import Review 

load_dotenv()


model = ChatOpenAI()

{
    "title": "Review",
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },      
}
}

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Hardware is great, but the software is terrible. The user interface is confusing and the app crashes frequently.""")

print(result)
print(result.summary)
print(result.sentiment)