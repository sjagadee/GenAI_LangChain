from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

# Create pipeline directly
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=512,
    temperature=0.7,
    device_map="auto"  # Automatically use GPU if available
)

# Wrap with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# llm = HuggingFacePipeline(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         temperature=0.5,
#         max_new_tokens=100
#     )
# )

model = ChatHuggingFace(llm=llm)
result = model.invoke("Which river is the longest river in USA")

print(result.content)