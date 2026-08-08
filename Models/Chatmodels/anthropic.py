from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv
load_dotenv()

model = ChatAnthropic(model_name="claude-3-opus" , temperature=0.7, max_completion_tokens=150)

result = model.invoke("Write a short story about a robot learning to love.")
print(result.content)