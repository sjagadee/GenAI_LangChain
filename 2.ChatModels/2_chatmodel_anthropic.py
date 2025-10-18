from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3-5-haiku-20241022")
result = model.invoke("Which river is the longest river in USA")

print(result.content)