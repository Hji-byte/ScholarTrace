from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents: list[Document], chunk_size: int = 900, chunk_overlap: int = 120) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        paper_id = chunk.metadata.get("paper_id", "unknown")
        chunk.metadata["chunk_id"] = f"{paper_id}-chunk-{index:04d}"
    return chunks

