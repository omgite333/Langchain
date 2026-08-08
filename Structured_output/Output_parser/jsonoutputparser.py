from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
     repo_id="Qwen/Qwen3-32B",
     task="text-generation"
)

model = ChatHuggingFace(llm=llm , max_output_tokens=100)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name,age and city of a fictional person \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
final_result = chain.invoke({})
print(final_result)