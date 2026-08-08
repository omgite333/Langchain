from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic'],
)

template2 = PromptTemplate(
    template='generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text'],
)

parser = StrOutputParser()

chain = RunnableSequence(template1, model, parser, template2, model, parser)

final_result = chain.invoke({'topic':'India'})

print(final_result)

chain.get_graph().print_ascii()