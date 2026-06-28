// Hybrid.swift
// Ported from graphrag-rs `retrieval::hybrid`.

import Foundation

/// Strategy used to merge ranked lists from different retrievers.
public enum FusionMethod: Sendable, Equatable {
    /// Reciprocal Rank Fusion (default).
    case rrf
    /// Weighted sum of max-normalized scores.
    case weighted
    /// Raw sum of scores.
    case combSum
    /// Maximum of the per-method scores.
    case maxScore
}

/// Configuration for `HybridRetriever`.
public struct HybridConfig: Sendable {
    public var semanticWeight: Float
    public var keywordWeight: Float
    public var fusionMethod: FusionMethod
    public var rrfK: Float
    public var maxCandidates: Int
    public var minScoreThreshold: Float

    public init(
        semanticWeight: Float = 0.7,
        keywordWeight: Float = 0.3,
        fusionMethod: FusionMethod = .rrf,
        rrfK: Float = 60.0,
        maxCandidates: Int = 100,
        minScoreThreshold: Float = 0.1
    ) {
        self.semanticWeight = semanticWeight
        self.keywordWeight = keywordWeight
        self.fusionMethod = fusionMethod
        self.rrfK = rrfK
        self.maxCandidates = maxCandidates
        self.minScoreThreshold = minScoreThreshold
    }
}

/// A fused search hit combining keyword and semantic signals.
public struct HybridSearchResult: Sendable, Equatable {
    public var id: String
    public var content: String
    public var score: Float
    public var semanticScore: Float
    public var keywordScore: Float

    public init(id: String, content: String, score: Float, semanticScore: Float, keywordScore: Float) {
        self.id = id
        self.content = content
        self.score = score
        self.semanticScore = semanticScore
        self.keywordScore = keywordScore
    }
}

/// Combines BM25 keyword search with cosine vector search over a chunk corpus.
public struct HybridRetriever: Sendable {
    public var config: HybridConfig
    private var bm25: BM25Retriever
    private var vectors: InMemoryVectorStore
    private var contents: [String: String] = [:]

    public init(config: HybridConfig = HybridConfig()) {
        self.config = config
        self.bm25 = BM25Retriever()
        self.vectors = InMemoryVectorStore()
    }

    public var isInitialized: Bool { !contents.isEmpty }
    public var documentCount: Int { contents.count }

    /// Index a chunk for keyword search, and for semantic search if it carries
    /// an embedding.
    public mutating func index(id: String, content: String, embedding: [Float]?) {
        contents[id] = content
        bm25.index(id: id, content: content)
        if let embedding {
            vectors.add(id: id, vector: embedding)
        } else {
            // Drop any vector from a previous version so semantic search can't
            // return this id using a stale embedding.
            vectors.remove(id: id)
        }
    }

    /// Index all chunks of a knowledge graph as a full (re)index. Clears any
    /// previously indexed content first, so ids removed since the last index
    /// can't linger in `contents`, BM25, or the vector store.
    public mutating func index(graph: KnowledgeGraph) {
        clear()
        for chunk in graph.chunks {
            index(id: chunk.id.raw, content: chunk.content, embedding: chunk.embedding)
        }
    }

    public mutating func clear() {
        bm25.clear()
        vectors.clear()
        contents.removeAll()
    }

    /// Run both retrievers and fuse the results.
    ///
    /// - Parameters:
    ///   - query: The raw query text (for BM25).
    ///   - queryEmbedding: Optional query vector (for semantic search).
    ///   - limit: Number of fused results to return.
    ///   - semanticThreshold: Minimum cosine similarity for a semantic hit.
    ///   - includeKeyword: Include BM25 results (false for a semantic-only approach).
    ///   - includeSemantic: Include vector results (false for a keyword-only approach).
    public func search(
        query: String,
        queryEmbedding: [Float]?,
        limit: Int,
        semanticThreshold: Float = 0,
        includeKeyword: Bool = true,
        includeSemantic: Bool = true
    ) -> [HybridSearchResult] {
        // A negative limit would trap in `prefix`; treat anything <= 0 as empty.
        guard limit > 0 else { return [] }
        let keyword: [(id: String, score: Float)] =
            includeKeyword
            ? bm25.search(query, limit: config.maxCandidates).map { (id: $0.id, score: $0.score) }
            : []
        // Drop non-positive cosine hits (off-topic protection: the vector store
        // always returns its nearest `maxCandidates`) and anything below the
        // caller's similarity threshold.
        let semantic: [(id: String, score: Float)] =
            (includeSemantic ? queryEmbedding : nil).map {
                vectors.search($0, k: config.maxCandidates)
                    .filter { $0.score > 0 && $0.score >= semanticThreshold }
            } ?? []

        let fused = fuse(semantic: semantic, keyword: keyword)
        // RRF scores are rank-based and inherently small (≈ 1/(k+rank)); the
        // absolute `minScoreThreshold` only makes sense for magnitude-based
        // fusion (weighted / CombSUM / MaxScore).
        let applyThreshold = config.fusionMethod != .rrf
        return Array(
            fused
                .filter { !applyThreshold || $0.score >= config.minScoreThreshold }
                .prefix(limit)
        )
    }

    // MARK: - Fusion

    private func fuse(
        semantic: [(id: String, score: Float)],
        keyword: [(id: String, score: Float)]
    ) -> [HybridSearchResult] {
        var semScore: [String: Float] = [:]
        var kwScore: [String: Float] = [:]
        var semRank: [String: Int] = [:]
        var kwRank: [String: Int] = [:]
        for (rank, item) in semantic.enumerated() {
            semScore[item.id] = item.score
            semRank[item.id] = rank
        }
        for (rank, item) in keyword.enumerated() {
            kwScore[item.id] = item.score
            kwRank[item.id] = rank
        }

        let maxSem = semantic.map(\.score).max() ?? 0
        let maxKw = keyword.map(\.score).max() ?? 0
        let allIDs = Set(semScore.keys).union(kwScore.keys)

        var results: [HybridSearchResult] = []
        for id in allIDs {
            let sem = semScore[id] ?? 0
            let kw = kwScore[id] ?? 0
            let combined: Float
            switch config.fusionMethod {
            case .rrf:
                var s: Float = 0
                if let r = semRank[id] {
                    s += (1.0 / (config.rrfK + Float(r) + 1.0)) * config.semanticWeight
                }
                if let r = kwRank[id] {
                    s += (1.0 / (config.rrfK + Float(r) + 1.0)) * config.keywordWeight
                }
                combined = s
            case .weighted:
                let nSem = maxSem > 0 ? sem / maxSem : 0
                let nKw = maxKw > 0 ? kw / maxKw : 0
                combined = nSem * config.semanticWeight + nKw * config.keywordWeight
            case .combSum:
                combined = sem + kw
            case .maxScore:
                combined = max(sem, kw)
            }
            results.append(
                HybridSearchResult(
                    id: id, content: contents[id] ?? "",
                    score: combined, semanticScore: sem, keywordScore: kw))
        }

        results.sort { lhs, rhs in
            if lhs.score == rhs.score { return lhs.id < rhs.id }
            return lhs.score > rhs.score
        }
        return results
    }
}

