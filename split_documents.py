
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load the PDF
loader = PyPDFLoader("course_notes.pdf")
documents = loader.load()

# Step 2: Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=[
        "\n\n",  # Prefer paragraph boundaries
        "\n",    # Then line boundaries
        ". ",    # Then sentence boundaries
        " ",     # Then word boundaries
        ""       # Finally, individual characters if needed
    ]
)

# Step 3: Split the documents
chunks = text_splitter.split_documents(documents)

# Step 4: Display the results
print("PDF loaded successfully!")
print("Total pages:", len(documents))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {i} ---")
    print("Page:", chunk.metadata.get("page"))
    print(chunk.page_content)
