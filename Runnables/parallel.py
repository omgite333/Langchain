from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel , RunnableSequence


llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

template1 = PromptTemplate(
    template='Generate a short and simple notes from the following topic: {text}}',
    input_variables=['text'],
)

template2 = PromptTemplate(
    template='generate 5 short question answeers from the following text \n {text}',
    input_variables=['text'],
)

template3 = PromptTemplate(
    template='Merge the provided notes and quiz into  single document \n notes -> {notes} \n quiz -> {quiz}',
    input_variables=['notes','quiz'],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes":RunnableSequence(template1, model1, parser),
    "quiz":RunnableSequence(template2, model2, parser)
})

merge_chain = RunnableSequence(template3, model1, parser)

final_chain = parallel_chain | merge_chain

final_result = final_chain.invoke({'topic':'India'})

print(final_result)

final_chain.get_graph().print_ascii()