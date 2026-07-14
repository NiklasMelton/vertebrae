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
Cosine retrieval requires every query and gallery embedding to have a nonzero L2 norm;
scoring fails with the affected endpoint and row identities when this requirement is
violated. Zero vectors remain valid for dot-product and squared-L2 retrieval.

Optional paired compression is fitted on gallery embeddings and applied to query
embeddings with the same transform. Requests whose `n_components` is greater than or
equal to the endpoint width are explicit no-ops: both endpoints retain their values,
sparsity, and dtype, and result metadata records `applied=False` with a warning. The
same rule applies in local and artifact-backed retrieval workflows.

The protocol is deliberately not an ANN, reranking, learned retrieval, or recommender
benchmark. It evaluates the fixed embedding geometry and declared candidate set only.
