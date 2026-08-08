from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, max_output_tokens=150)

result = model.invoke("Write a short story about a robot learning to love.")
print(result.content)