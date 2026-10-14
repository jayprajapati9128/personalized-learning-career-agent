
from langchain_community.document_loaders import PyPDFLoader

# Load the course PDF
loader = PyPDFLoader("course_notes.pdf")

# Extract pages from the PDF
documents = loader.load()

# Display the results
print("PDF loaded successfully!")
print("Total pages:", len(documents))

if documents:
    print("\nFirst page content:\n")
    print(documents[0].page_content[:1000])
