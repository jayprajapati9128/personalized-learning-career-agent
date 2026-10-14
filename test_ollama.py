
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3.5:0.8b",
    temperature=0,
    num_predict=100,
    reasoning=False
)

print("Testing Ollama...")

response = llm.invoke(
    "Explain RAG in two simple sentences."
)

print(response.content)
