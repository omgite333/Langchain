from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
     repo_id="Qwen/Qwen3-32B",
     task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="write a 5 line summary on the following text . /n {text} ",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic": "Black hole"})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result.content})

result2 = model.invoke(prompt2)
print(result2.content)