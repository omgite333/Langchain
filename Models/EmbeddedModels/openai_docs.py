from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=100)

document = ["The capital of France is Paris."
            "what is the capital of France?" ]

result = embeddings.embed_documents([document])
print(string(result))