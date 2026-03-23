import uuid
from datetime import datetime, timezone
import chromadb

CLIENT = chromadb.PersistentClient(path="./chroma_db")
COLLECTION = CLIENT.get_or_create_collection(
    name="memories",
    embedding_function=None
)

def remember(content, embedding, time):
    """
    content: str
    embedding: list[float]
    returns: id str
    """
    if not isinstance(content, str):
        raise TypeError("content must be a str")
    if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
        raise TypeError("embedding must be a list of floats")

    item_id = str(uuid.uuid4())
    COLLECTION.add(
        ids=[item_id],
        documents=[content],
        embeddings=[embedding],
        metadatas=[{"time": time}],
    )
    return item_id

def query(query_embedding, k):
    """
    query_embedding: list[float]
    returns: list of dicts, best first
    """
    if not isinstance(query_embedding, list) or not all(isinstance(x, (int, float)) for x in query_embedding):
        raise TypeError("query_embedding must be a list of floats")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be > 0")

    res = COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    ids = res["ids"][0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out = []
    for i in range(len(ids)):
        t = metas[i].get("time") if metas[i] else None
        c = docs[i]
        out.append([t, c])
    return out

#temporarily for mettaclaw:
def import_ltm_file(filename):
    if not isinstance(filename, str):
        raise TypeError("filename must be a str")
    inserted_ids = []
    with open(filename, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                time_str = line.split('("', 1)[1].split('" "', 1)[0]
                content_start = line.find('" "') + 3
                content_end = line.rfind('" (')
                content_str = line[content_start:content_end]
                vector_str = line[content_end + 3:]
                vector_str = vector_str[:-2].strip()   # remove final ))
                embedding = [float(x) for x in vector_str.split()] if vector_str else []
                #inserted_ids.append((content_str, embedding, time_str))
                remember(content_str, embedding, time_str)
            except Exception as e:
                raise ValueError(f"parse error in {filename}:{lineno}: {e}") from e
    return inserted_ids
