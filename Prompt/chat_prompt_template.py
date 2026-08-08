from langchain_core.prompts import PromptTemplate

chat_template = PromptTemplate(
    ('system', "You are a helpful {Domain} expert"),
    ('human', "Explain me in simple terms, what is {topic}?")
)

prompt = chat_template.invoke({
    'Domain': 'AI',
    'topic': 'ChatGPT'
})

print(prompt)