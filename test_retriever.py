
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Step 1: Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Load the existing Chroma database
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Step 3: Create the retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# Step 4: Test two types of questions
questions = [
    "What is RAG?",
    "How does a RAG-based AI study companion help students learn and find career opportunities?"
]

# Step 5: Display retrieved results
for query in questions:
    print("\n" + "=" * 60)
    print("Question:", query)

    results = retriever.invoke(query)

    print("Retrieved results:", len(results))

    for i, document in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(document.page_content[:700])
