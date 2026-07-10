# retrieval and matching

`vertebrae` can compare frozen embedding backbones under an explicit exact
query--gallery ranking protocol. This is complementary to OverlapIndex:
OverlapIndex measures global labeled representation separation, while retrieval
measures whether declared relevant candidates rank near the top for each query.

Use `RetrievalDataset` to provide independent query and gallery inputs plus either a
dense grade matrix or sparse `(query_id, gallery_id, grade)` records. Relevance grades
must be finite and non-negative; grades above zero are relevant for binary metrics.
Missing pairs have grade zero. Equal query/gallery IDs are valid matches unless you
explicitly place them in `exclusions`.

`RetrievalBenchmark` supports precomputed embeddings, ordinary same-modality
extractors, and explicit branch encoders through `CallableRetrievalExtractor` or the
retrieval-capable OpenCLIP, SigLIP, and compatible Hugging Face multimodal adapters.
The default is exact cosine ranking with NDCG@10 as the primary score. Reports include
NDCG, precision, recall, hit rate, MRR, mAP, and positive/negative similarity margins.

The protocol is deliberately not an ANN, reranking, learned retrieval, or recommender
benchmark. It evaluates the fixed embedding geometry and declared candidate set only.
