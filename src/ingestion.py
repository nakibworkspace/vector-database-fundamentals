import os
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# 1. Setup
client = QdrantClient("http://localhost:6333")
collection_name = "pdf_knowledge_base"
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create Collection
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

def ingest_pdf(pdf_path):
    # 2. Extract Text
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # 3. Chunking (Simple splitting by length)
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    
    # 4. Embed and Upload
    points = []
    for i, chunk in enumerate(chunks):
        vector = model.encode(chunk).tolist()
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk, "page": i+1}
            )
        )
    
    client.upsert(collection_name=collection_name, points=points)
    print(f"Ingested {len(chunks)} chunks from {pdf_path}")

if __name__ == "__main__":
    # Ensure a pdf exists in data/sample.pdf
    ingest_pdf("data/sample.pdf")