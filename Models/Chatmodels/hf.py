from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
     repo_id="Qwen/Qwen3-32B",
     task="text-generation"
)

model = ChatHuggingFace(llm=llm, temperature=0.7, max_completion_tokens=150)

result = model.invoke("What is the capital of France?")
print(result.content)