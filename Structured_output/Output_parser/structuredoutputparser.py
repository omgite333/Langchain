from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, RespoceSchema

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
     repo_id="Qwen/Qwen3-32B",
     task="text-generation"
)

model = ChatHuggingFace(llm=llm , max_output_tokens=100)

schema = [
    RespoceSchema(name="fact_1", description="fact 1 about the topic"),
    RespoceSchema(name="fact_2", description="fact 2 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me 2 facts about {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"topic": "Black hole"})

print(result)