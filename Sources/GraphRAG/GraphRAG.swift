// GraphRAG.swift
// Umbrella documentation for the GraphRAG Swift package — a port of the Rust
// crate graphrag-rs (https://github.com/automataIA/graphrag-rs).
//
// GraphRAG builds a knowledge graph from documents and answers natural-language
// questions using graph-based context retrieval.
//
// Quick start:
// ```swift
// import GraphRAG
//
// let rag = try GraphRAGBuilder()
//     .withChunkSize(800)
//     .withChunkOverlap(100)
//     .withTopK(5)
//     .build()
//
// await rag.addDocument(text: "Ada Lovelace worked with Charles Babbage ...")
// try await rag.build()
// let answer = try await rag.ask("Who did Ada Lovelace work with?")
// print(answer.text)
// ```
//
// Everything in this package is `public`. The principal entry points are:
//   - `GraphRAG`            the orchestrating actor (ingest → build → ask)
//   - `GraphRAGBuilder`     fluent configuration
//   - `Config`              tunable defaults
//   - `KnowledgeGraph`      the entity/relationship graph + documents/chunks
//   - `HybridRetriever`     BM25 + vector fusion retrieval
//   - `PageRank`, `GraphTraversal`, `GraphAnalytics`  graph algorithms
//
// Pluggable backends conform to `EmbeddingModel`, `LanguageModel`, and
// `EntityExtracting`. Offline defaults (`HashEmbedder`, `PatternEntityExtractor`)
// require no network or model download; `OllamaClient` / `OllamaEmbedder` enable
// local LLM-backed extraction and generation.

/// The semantic version of this GraphRAG port.
public enum GraphRAGVersion {
    public static let current = "0.2.0"
}
