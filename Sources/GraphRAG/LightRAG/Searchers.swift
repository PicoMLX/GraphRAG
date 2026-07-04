// Searchers.swift
// A concrete `SemanticSearcher` (BM25 + cosine over an in-memory corpus) and
// builders that turn a knowledge graph into the two LightRAG stores.

import Foundation

/// A `SemanticSearcher` backed by a `HybridRetriever` over a fixed corpus.
///
/// Each indexed document carries the real chunk ids it stands for
/// (`sourceChunks`), so a high-level community hit still resolves back to the
/// actual graph chunks that ground it rather than to its synthetic id.
public struct InMemorySemanticSearcher: SemanticSearcher {
    private let retriever: HybridRetriever
    private let embedder: any EmbeddingModel
    private let sourceChunksByID: [String: [String]]

    private init(
        retriever: HybridRetriever, embedder: any EmbeddingModel,
        sourceChunksByID: [String: [String]]
    ) {
        self.retriever = retriever
        self.embedder = embedder
        self.sourceChunksByID = sourceChunksByID
    }

    /// Build a searcher, reusing each document's precomputed embedding when one
    /// is supplied and embedding on demand otherwise. `sourceChunks` records the
    /// real chunk ids each document grounds to.
    public static func build(
        documents: [(id: String, content: String, embedding: [Float]?, sourceChunks: [String])],
        embedder: any EmbeddingModel
    ) async throws -> InMemorySemanticSearcher {
        // Let the candidate pool cover the whole corpus so a large `topK` isn't
        // silently capped below the number of available documents.
        var retriever = HybridRetriever(
            config: HybridConfig(maxCandidates: max(100, documents.count)))
        var sourceChunksByID: [String: [String]] = [:]
        for doc in documents {
            // Reuse a precomputed vector only when its dimension matches this
            // embedder — a mismatch (e.g. a graph saved under a different
            // embedder) would make cosine similarity silently return 0. Branch
            // explicitly since `??` can't wrap an async default (its autoclosure
            // isn't async).
            let embedding: [Float]
            if let precomputed = doc.embedding, precomputed.count == embedder.dimension {
                embedding = precomputed
            } else {
                embedding = try await embedder.embed(doc.content)
            }
            retriever.index(id: doc.id, content: doc.content, embedding: embedding)
            sourceChunksByID[doc.id] = doc.sourceChunks
        }
        return InMemorySemanticSearcher(
            retriever: retriever, embedder: embedder, sourceChunksByID: sourceChunksByID)
    }

    public func search(_ query: String, topK: Int) async throws -> [LightRAGResult] {
        // Design note: LightRAG is a self-contained retrieval subsystem with its
        // own configuration (DualRetrievalConfig / KeywordExtractorConfig). It
        // uses full hybrid (BM25 + cosine) search with the default similarity
        // threshold on purpose and does NOT inherit the owning GraphRAG's
        // hybrid-path Config (`approach`, `similarityThreshold`) — those govern
        // `GraphRAG.ask`/`search`, a different retrieval path. This keeps the two
        // strategies independent; a caller who wants semantic-only or a stricter
        // threshold uses the GraphRAG path (or we can thread those settings in if
        // that inheritance is desired).
        let queryEmbedding = try await embedder.embed(query)
        let hits = retriever.search(query: query, queryEmbedding: queryEmbedding, limit: topK)
        return hits.map {
            LightRAGResult(
                id: $0.id, content: $0.content, score: $0.score,
                sourceChunks: sourceChunksByID[$0.id] ?? [$0.id])
        }
    }
}

/// Builders for the LightRAG stores from a `KnowledgeGraph`.
public enum LightRAG {
    /// Low-level (entity/detail) store: the document chunks themselves. Each
    /// chunk grounds to itself.
    public static func chunkSearcher(
        graph: KnowledgeGraph, embedder: any EmbeddingModel
    ) async throws -> InMemorySemanticSearcher {
        // Reuse the embeddings computed during the graph build, when present.
        let documents = graph.chunks.map {
            (id: $0.id.raw, content: $0.content, embedding: $0.embedding, sourceChunks: [$0.id.raw])
        }
        return try await InMemorySemanticSearcher.build(documents: documents, embedder: embedder)
    }

    /// High-level (theme/global) store: one short summary per detected community.
    /// A community grounds to the real chunks that mention its member entities,
    /// so answers built from high-level hits still cite actual evidence chunks.
    public static func communitySearcher(
        graph: KnowledgeGraph, communities: CommunityDetectionResult, embedder: any EmbeddingModel
    ) async throws -> InMemorySemanticSearcher {
        // Summaries are derived text with no precomputed embedding — embed on demand.
        let documents = communities.communities.map { community in
            (
                id: "community_\(community.id)",
                content: communitySummary(community, graph: graph),
                embedding: [Float]?.none,
                sourceChunks: communitySourceChunks(community, graph: graph)
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

    /// The real chunk ids that ground a community's member entities, deterministic
    /// and deduplicated in first-seen order. These are the evidence for a
    /// high-level community hit.
    ///
    /// Uses every evidence representation a `KnowledgeGraph` may carry: chunks
    /// annotated with member entities (`TextChunk.entities`), the members' own
    /// mention evidence (`Entity.mentions`), and the context chunks of the
    /// relationships connecting members (`Relationship.context`). The build
    /// pipeline populates chunk annotations, but externally-constructed graphs
    /// may carry only mentions or only relationship context — any one alone still
    /// yields grounding.
    public static func communitySourceChunks(
        _ community: Community, graph: KnowledgeGraph
    ) -> [String] {
        let memberSet = Set(community.members)
        var ids: [String] = []
        var seen: Set<String> = []
        func add(_ id: String) {
            if seen.insert(id).inserted { ids.append(id) }
        }
        // Chunks annotated with a member entity, in graph chunk order.
        for chunk in graph.chunks where chunk.entities.contains(where: memberSet.contains) {
            add(chunk.id.raw)
        }
        // Member entities' own mention evidence, in member then mention order.
        for member in community.members {
            guard let entity = graph.entity(member) else { continue }
            for mention in entity.mentions { add(mention.chunkID.raw) }
        }
        // Context chunks of relationships internal to the community — the
        // evidence behind the relationship types the summary is built from.
        for relationship in graph.relationships
        where memberSet.contains(relationship.source) && memberSet.contains(relationship.target) {
            for chunkID in relationship.context { add(chunkID.raw) }
        }
        return ids
    }
}
