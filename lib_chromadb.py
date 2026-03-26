import uuid
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

def forget_ids(item_ids):
    """
    item_ids: list[str]
    deletes items by ids
    returns: deleted ids
    """
    if not isinstance(item_ids, list) or not all(isinstance(x, str) for x in item_ids):
        raise TypeError("item_ids must be a list of str")
    if not item_ids:
        return []
    COLLECTION.delete(ids=item_ids)
    return item_ids

def forget_id(item_id):
    """
    item_id: str
    deletes one item by id
    returns: deleted id
    """
    if not isinstance(item_id, str):
        raise TypeError("item_id must be a str")
    forget_ids([item_id])
    return item_id

def query(query_embedding, k):
    """
    query_embedding: list[float]
    returns: list of [time, content]
    """
    res = query_with_ids(query_embedding, k)
    return [[t, c] for _, t, c in res]

def query_with_ids(query_embedding, k):
    """
    query_embedding: list[float]
    returns: list of items
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
        out.append([ids[i], t, c])
    return out
