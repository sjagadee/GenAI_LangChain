from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "Virat Kholi is an Indian cricket player known for his aggressive batting style and leadership skills.",
    "Rohit Sharma is a right-handed batsman who has been a key player for India and record breaking double century.",
    "MS Dhoni is an Indian cricketer who is widely considered to be the best captain in the history of Indian cricket.",
    "Sachin Tendulkar is an Indian cricketer who is widely regarded as one of the greatest batsmen of all time.",
    "Jasprit Bumrah is an Indian cricketer who is the best fast bowler India has ever had and has a unique bowling style.",
]

query = "Who is the all time best captain of India?"

embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarities = cosine_similarity([query_embedding],embeddings)[0]
print(similarities)

indices = np.argsort(similarities)[::-1]
print(indices)

print(documents[indices[0]])
