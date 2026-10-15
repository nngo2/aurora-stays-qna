# Retrieval module shared by the notebook and the backend (single source of truth): this cell writes the file,
# the next cell imports it, and the backend section copies the same file.
# Hybrid search = semantic search (FAISS) + keyword search (BM25), merged with reciprocal rank fusion,
# with the property code added to questions that name a hotel by its city or name.
import re
from rank_bm25 import BM25Okapi
from aurora_properties import build_property_map, add_property_codes

# Common words carry no policy meaning; removing them keeps the keyword scores focused on terms such as
# property codes, room types, and rate plans
STOPWORDS = set("""a an and are as at be booked by can could do does for from had has have how i if in is it its
my of on or our should so that the their there they this to until was we what when where which who will with
would you your""".split())

def tokenize(text):
    # Lowercase words; keep codes and times such as "ash-sea" and "6:00" as one token, and also add the parts
    # of hyphenated words ("ash-sea" -> "ash-sea", "ash", "sea"), so "check in" still matches "check-in"
    tokens = []
    for t in re.findall(r"[a-z0-9]+(?:[-:.][a-z0-9]+)*", text.lower()):
        if t in STOPWORDS:
            continue
        tokens.append(t)
        if "-" in t:
            tokens.extend(t.split("-"))
    return tokens

class HybridRetriever:
    # Ranks every chunk twice, by meaning (FAISS) and by keywords (BM25), and merges the two rankings with
    # reciprocal rank fusion (RRF): score = sum of 1 / (RRF_K + rank). A chunk ranked high by either method
    # rises to the top, and no score scaling is needed because only the ranks are used.
    RRF_K = 60  # standard constant from the RRF paper; softens the gap between the very top ranks

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        # The chunks in index order, read from the FAISS docstore, so the backend needs no extra file
        self.docs = [
            vectorstore.docstore.search(vectorstore.index_to_docstore_id[i])
            for i in range(vectorstore.index.ntotal)
        ]
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs])
        self.properties = build_property_map(d.page_content for d in self.docs)  # code -> (name, city, state)

    def search(self, query, k):
        # "the Denver hotel" -> "the Denver hotel (ASH-DEN Union)", so both methods see the property code
        query = add_property_codes(query, self.properties)

        # Full ranking from each method (the whole index is small, so ranking every chunk is cheap)
        semantic = self.vectorstore.similarity_search(query, k=len(self.docs))
        scores = self.bm25.get_scores(tokenize(query))
        keyword = [self.docs[i] for i in sorted(range(len(self.docs)), key=lambda i: -scores[i]) if scores[i] > 0]

        fused = {}
        for ranking in (semantic, keyword):
            for rank, doc in enumerate(ranking, start=1):
                entry = fused.setdefault(doc.page_content, [0.0, doc])  # chunk text is unique (duplicates were removed)
                entry[0] += 1 / (self.RRF_K + rank)
        return [doc for _, doc in sorted(fused.values(), key=lambda e: -e[0])][:k]

_retrievers = {}  # one HybridRetriever per vector store, built on first use

def get_retriever(vectorstore):
    # The HybridRetriever for this vector store (built once, then reused)
    cached = _retrievers.get(id(vectorstore))
    if cached is None or cached.vectorstore is not vectorstore:
        cached = _retrievers[id(vectorstore)] = HybridRetriever(vectorstore)
    return cached

def retrieve(query: str, k: int = 2, vectorstore=None) -> list[str]:
    # Top-k chunks from hybrid search
    results = get_retriever(vectorstore).search(query, k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]
