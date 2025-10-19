from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# result = embedding.embed_query("Austin is the capital of Texas")

docs = [
    "Austin is the capital of Texas",
    "Springfield is the capital of Illinois",
    "Sacramento is the capital of california"
]

result = embedding.embed_documents(docs)

print(str(result))