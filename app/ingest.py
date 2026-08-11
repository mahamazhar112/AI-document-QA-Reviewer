from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

from app.pdf_parser import parse_folder
from app.metadata import infer_source_type

DATA_DIR = Path("data/reference_docs")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "qa_reference_docs"


def build_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    sections = parse_folder(str(DATA_DIR))

    documents, metadatas, ids = [], [], []
    for i, section in enumerate(sections):
        if section.main_heading.strip().lower() == "contents":
            continue

        chunk_text = f"{section.main_heading} - {section.sub_heading}: {section.text}".strip(" -:")
        documents.append(chunk_text)
        metadatas.append({
            "source_type": infer_source_type(section.source_file),
            "source_file": section.source_file,
            "main_heading": section.main_heading,
            "sub_heading": section.sub_heading,
        })
        ids.append(f"chunk_{i}")

    embeddings = model.encode(documents, show_progress_bar=True).tolist()
    collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    print(f"Indexed {len(documents)} chunks into '{COLLECTION_NAME}'.")
    return collection


if __name__ == "__main__":
    build_index()