from langchain_openai import ChaOpenAI, ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated

load_dotenv()


model = ChatOpenAI()

class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment:Annotated[str, "The sentiment of the review, either positive or negative"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Hardware is great, but the software is terrible. The user interface is confusing and the app crashes frequently.""")

print(result)
print(result['summary'])
print(result['sentiment'])