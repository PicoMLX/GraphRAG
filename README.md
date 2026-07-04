# GraphRAG (Swift)

[![CI](https://github.com/PicoMLX/GraphRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/PicoMLX/GraphRAG/actions/workflows/ci.yml)

A Swift port of the Rust crate [`graphrag-rs`](https://github.com/automataIA/graphrag-rs):
Graph-based Retrieval Augmented Generation. It builds a knowledge graph from
documents and answers natural-language questions using graph-based context
retrieval.

This package ports the **core library** (`graphrag-core`) — the parts that make
GraphRAG work end to end — into idiomatic, Swift 6, dependency-free code. It runs
fully offline out of the box, and can optionally talk to a local
[Ollama](https://ollama.com) server for LLM-backed extraction and answer
generation.

## Installation

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/picomlx/graphrag.git", branch: "main")
```

and depend on the `GraphRAG` product.

## Quick start

```swift
import GraphRAG

// Offline pipeline: hash embeddings + pattern-based entity extraction.
let rag = try GraphRAGBuilder()
    .withChunkSize(800)
    .withChunkOverlap(100)
    .withTopK(5)
    .build()

await rag.addDocument(text: """
    Ada Lovelace collaborated with Charles Babbage on the Analytical Engine,
    an early mechanical general-purpose computer.
    """)

try await rag.build()                       // chunk → extract → embed → index
let answer = try await rag.ask("Who worked on the Analytical Engine?")
print(answer.text)
print(answer.sources)                        // grounding chunk ids
```

### Using a local LLM (Ollama)

```swift
let rag = try GraphRAGBuilder()
    .withLocalDefaults()                      // Ollama chat + embeddings
    .build()
```

With Ollama enabled, entity/relationship extraction uses the LLM extraction
prompt, and `ask` synthesizes a natural-language answer from the retrieved
context. Without it, extraction is pattern-based and `ask` returns an extractive
summary of the top chunks.

## What's included

| Area | Types |
| --- | --- |
| Core model | `Document`, `TextChunk`, `Entity`, `Relationship`, `EntityMention`, typed IDs, `GraphRAGError` |
| Abstractions | `LanguageModel`, `EmbeddingModel`, `EntityExtracting`, `ChunkingStrategy` |
| Text | `HierarchicalChunker`, `TextProcessor`, `TfIdfKeywordExtractor` |
| Graph | `KnowledgeGraph`, `PageRank`, `GraphTraversal` (BFS/DFS/ego/paths), `GraphAnalytics` (degree/closeness/betweenness/components) |
| Retrieval | `BM25Retriever`, `InMemoryVectorStore` (cosine), `HybridRetriever` (RRF / weighted / CombSUM / MaxScore fusion) |
| Extraction | `PatternEntityExtractor`, `LLMEntityExtractor`, `Prompts` |
| Embeddings | `HashEmbedder` (offline, deterministic), `OllamaEmbedder` |
| LLM | `OllamaClient` |
| Communities | `LeidenCommunityDetector` (weighted, deterministic), `Community` |
| LightRAG | `LightRAGEngine`, `DualLevelRetriever`, `KeywordExtractor`, `SemanticSearcher` |
| Orchestration | `GraphRAG` (actor), `GraphRAGBuilder`, `Config` |

## Design notes / port fidelity

- **Defaults match the Rust crate**: PageRank damping `0.85` / tolerance `1e-6`,
  BM25 `k1 = 1.2`, `b = 0.75`, hybrid `RRF k = 60`, semantic/keyword weights
  `0.7 / 0.3`, traversal `maxDepth = 3`, min relationship strength `0.5`, etc.
- **Concurrency**: `GraphRAG` is an `actor`; backends are `Sendable` existentials
  (`any EmbeddingModel`, `any LanguageModel`, `any EntityExtracting`). Builds
  cleanly under Swift 6 strict concurrency.
- **Unicode safety**: the Rust chunker works on UTF-8 byte offsets guarded by
  `is_char_boundary`. This port operates on `Character` (grapheme) arrays, which
  are always valid boundaries; sizes and offsets are measured in characters.
- **Scope**: this is the portable core pipeline plus the LightRAG dual-level
  retrieval and Leiden community-detection subsystems (see below). The Rust
  workspace's server/WASM/CLI crates and other optional subsystems (ROGRAG,
  distributed caching, persistence backends) remain out of scope for this port.

## Community detection (Leiden)

`LeidenCommunityDetector` partitions the knowledge graph into communities via
greedy modularity local-moving plus a refinement pass that splits internally
disconnected communities. It ports the structure of the Rust crate's
single-level Leiden, but makes it **deterministic** (stable node ordering, so
repeated runs give identical assignments) and **weighted** — it uses each
relationship's `confidence` as an edge weight, which the Rust version ignored.

```swift
let graph = await rag.knowledgeGraph()
let result = LeidenCommunityDetector().detect(graph)
for community in result.communities {
    print("community \(community.id): \(community.members.count) members")
}
print("modularity:", result.modularity)   // Newman modularity of the partition
```

Only knobs that affect the result are exposed via `LeidenConfig`: `resolution`
(higher → more, smaller communities), `maxIterations`, and `minModularityGain`.

## Dual-level retrieval (LightRAG)

`LightRAGEngine` answers queries by searching two levels at once: a **low-level**
store over document chunks (entity/detail-centric) and a **high-level** store
over per-community theme summaries derived from Leiden (global/relationship-
centric). A `KeywordExtractor` splits the query into high- and low-level
keywords using the LLM (with a deterministic offline fallback), each level is
searched independently, and the hits are merged — `interleave` (default),
`highFirst`, `lowFirst`, or `weighted`.

```swift
let engine = try await rag.lightRAG()   // requires a successful build() first
let answer = try await engine.ask("Who worked on the Analytical Engine?")
print(answer.text)

// Or inspect both levels directly:
let results = try await engine.retrieve("...", topK: 10)
print(results.highLevelChunks, results.lowLevelChunks, results.mergedChunks)
```

An engine created via `rag.lightRAG()` inherits that instance's retrieval
settings from `Config`, so it stays consistent with `rag.ask`/`rag.search`: it
honors `topKResults` (the default when `retrieve`/`ask` are called without an
explicit `topK`), applies `similarityThreshold` to semantic hits, and drops BM25
when `approach == "semantic"`. `approach == "keyword"` is rejected — LightRAG's
dual-level design requires embeddings. Constructing `LightRAGEngine` directly
uses permissive defaults (full hybrid, no threshold, `topK` 10).

## Testing

```bash
swift test
```

The suite covers chunking, keyword extraction, BM25 ranking, cosine/vector
search, the knowledge graph, PageRank, traversal, analytics, pattern extraction,
Leiden community detection, LightRAG dual-level retrieval, and the end-to-end
offline build/ask pipeline.
