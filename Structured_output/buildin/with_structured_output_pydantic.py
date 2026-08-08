from langchain_openai import ChaOpenAI, ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel 

load_dotenv()


model = ChatOpenAI()

class Review(BaseModel):
    summary:str
    sentiment:str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Hardware is great, but the software is terrible. The user interface is confusing and the app crashes frequently.""")

print(result)
print(result.summary)
print(result.sentiment)