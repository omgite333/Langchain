from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough , RunnableLambda , RunnableBranch

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic'],
)

prompt1 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text'],
)

def word_count(text):
    return len(text.split())


parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1 , model , parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>500 , RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain_gen_chain , branch_chain_chain)

print(final_chain.invoke({'topic':'Cricket'}))
