// Searchers.swift
// A concrete `SemanticSearcher` (BM25 + cosine over an in-memory corpus) and
// builders that turn a knowledge graph into the two LightRAG stores.

import Foundation

/// A `SemanticSearcher` backed by a `HybridRetriever` over a fixed corpus.
public struct InMemorySemanticSearcher: SemanticSearcher {
    private let retriever: HybridRetriever
    private let embedder: any EmbeddingModel

    private init(retriever: HybridRetriever, embedder: any EmbeddingModel) {
        self.retriever = retriever
        self.embedder = embedder
    }

    /// Build a searcher, reusing each document's precomputed embedding when one
    /// is supplied and embedding on demand otherwise.
    public static func build(
        documents: [(id: String, content: String, embedding: [Float]?)],
        embedder: any EmbeddingModel
    ) async throws -> InMemorySemanticSearcher {
        var retriever = HybridRetriever()
        for doc in documents {
            // `??` can't wrap an async default (its autoclosure isn't async), so
            // branch explicitly to embed only when no precomputed vector exists.
            let embedding: [Float]
            if let precomputed = doc.embedding {
                embedding = precomputed
            } else {
                embedding = try await embedder.embed(doc.content)
            }
            retriever.index(id: doc.id, content: doc.content, embedding: embedding)
        }
        return InMemorySemanticSearcher(retriever: retriever, embedder: embedder)
    }

    public func search(_ query: String, topK: Int) async throws -> [LightRAGResult] {
        let queryEmbedding = try await embedder.embed(query)
        let hits = retriever.search(query: query, queryEmbedding: queryEmbedding, limit: topK)
        return hits.map {
            LightRAGResult(id: $0.id, content: $0.content, score: $0.score, sourceChunks: [$0.id])
        }
    }
}

/// Builders for the LightRAG stores from a `KnowledgeGraph`.
public enum LightRAG {
    /// Low-level (entity/detail) store: the document chunks themselves.
    public static func chunkSearcher(
        graph: KnowledgeGraph, embedder: any EmbeddingModel
    ) async throws -> InMemorySemanticSearcher {
        // Reuse the embeddings computed during the graph build, when present.
        let documents = graph.chunks.map {
            (id: $0.id.raw, content: $0.content, embedding: $0.embedding)
        }
        return try await InMemorySemanticSearcher.build(documents: documents, embedder: embedder)
    }

    /// High-level (theme/global) store: one short summary per detected community.
    public static func communitySearcher(
        graph: KnowledgeGraph, communities: CommunityDetectionResult, embedder: any EmbeddingModel
    ) async throws -> InMemorySemanticSearcher {
        // Summaries are derived text with no precomputed embedding — embed on demand.
        let documents = communities.communities.map { community in
            (
                id: "community_\(community.id)",
                content: communitySummary(community, graph: graph),
                embedding: [Float]?.none
            )
        }
        return try await InMemorySemanticSearcher.build(documents: documents, embedder: embedder)
    }

    /// A short textual theme for a community: member names plus the relationship
    /// types connecting them.
    public static func communitySummary(_ community: Community, graph: KnowledgeGraph) -> String {
        let names = community.members.compactMap { graph.entity($0)?.name }
        let memberSet = Set(community.members)
        var relationTypes: Set<String> = []
        for member in community.members {
            for (neighbor, relationship) in graph.neighbors(of: member)
            where memberSet.contains(neighbor) {
                relationTypes.insert(relationship.relationType)
            }
        }
        var parts: [String] = []
        if !names.isEmpty { parts.append("Entities: " + names.joined(separator: ", ")) }
        if !relationTypes.isEmpty {
            parts.append("Relationships: " + relationTypes.sorted().joined(separator: ", "))
        }
        return parts.joined(separator: ". ")
    }
}
