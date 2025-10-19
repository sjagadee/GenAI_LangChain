from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

docs = [
    "Austin is the capital of Texas",
    "Springfield is the capital of Illinois",
    "Sacramento is the capital of california"
]

result = embedding.embed_documents(docs)
print(str(result))