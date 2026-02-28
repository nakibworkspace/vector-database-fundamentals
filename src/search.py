from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient("http://localhost:6333")
model = SentenceTransformer('all-MiniLM-L6-v2')
collection_name = "pdf_knowledge_base"

def search_knowledge(query):
    query_vector = model.encode(query).tolist()
    
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=2 # Top 2 results
    )
    
    print(f"\nQuery: {query}")
    for result in search_result:
        print(f"Score: {result.score:.4f} | Text: {result.payload['text'][:100]}...")

if __name__ == "__main__":
    search_knowledge("What is the main topic of the document?")