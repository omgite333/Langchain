from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic'],
)

template2 = PromptTemplate(
    template='explain the following joke -{text}',
    input_variables=['text'],
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(template1 , model , parser)

parralle_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explaination' :RunnableSequence(template2 , model, parser)
    })

final_chain = RunnableSequence(joke_gen_chain , parralle_chain)

print(final_chain.invoke({'topic':'Cricket'}))
