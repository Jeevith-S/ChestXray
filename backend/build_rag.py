# ==========================================
# LOAD PDF
# ==========================================

from pypdf import PdfReader

# Give your PDF path here
pdf_path = r"D:\chestxrayproject\Chest_Diseases_Medical_Reference.pdf"

# Load PDF
reader = PdfReader(pdf_path)

print("Total Pages:", len(reader.pages))

# ==========================================
# EXTRACT TEXT FROM PDF
# ==========================================

full_text = ""

for page in reader.pages:

    text = page.extract_text()

    if text:

        full_text += text + "\n"

print("Total Characters:", len(full_text))

# Optional: print first 1000 characters

print("\nSample Text:\n")

print(full_text[:1000])



# ==========================================
# MANUAL CHUNKING
# ==========================================

chunk_size = 500      # characters per chunk
overlap = 100         # overlap between chunks

chunks = []

start = 0

while start < len(full_text):

    end = start + chunk_size

    chunk = full_text[start:end]

    chunks.append(chunk)

    start += (chunk_size - overlap)

print("\nTotal Chunks:", len(chunks))

# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("\nEmbedding Model Loaded")

# ==========================================
# CREATE EMBEDDINGS
# ==========================================

embeddings = embedding_model.encode(

    chunks,

    show_progress_bar=True

)

print("\nEmbedding Shape:")

print(embeddings.shape)

embeddings = embedding_model.encode(
    chunks,
    show_progress_bar=True
)
# ==========================================
# BUILD FAISS INDEX
# ==========================================

import faiss
import numpy as np

# Convert to float32 because FAISS requires float32

embeddings = np.array(
    embeddings
).astype("float32")

print("Embedding Shape:",
      embeddings.shape)

# Get vector dimension

dimension = embeddings.shape[1]

print("Vector Dimension:",
      dimension)

# Create FAISS Index

index = faiss.IndexFlatL2(
    dimension
)

# Store all vectors

index.add(embeddings)

print("Total Vectors Stored:",
      index.ntotal)
# ==========================================
# SAVE FAISS INDEX
# ==========================================

faiss.write_index(
    index,
    "medical_index.faiss"
)

print(
    "\n✅ FAISS Index Saved"
)
# ==========================================
# SAVE CHUNKS
# ==========================================

import pickle

with open(
    "medical_chunks.pkl",
    "wb"
) as f:

    pickle.dump(
        chunks,
        f
    )

print(
    "✅ Chunks Saved"
)
