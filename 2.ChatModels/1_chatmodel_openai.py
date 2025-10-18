from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# model = ChatOpenAI(model="gpt-4", temparature=0)
# result = model.invoke("What is the capital of USA")

model = ChatOpenAI(model="gpt-4", temperature=1.5, max_completion_tokens=20)
result = model.invoke("Write a 5 line poem, about forest")

print("Result", result.content)