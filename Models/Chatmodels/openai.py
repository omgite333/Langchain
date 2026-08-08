from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model_name="gpt-4" , temperature=0.7, max_completion_tokens=150)

result = model.invoke("Write a short story about a robot learning to love.")
print(result.content)