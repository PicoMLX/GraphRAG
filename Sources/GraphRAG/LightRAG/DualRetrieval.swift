// DualRetrieval.swift
// Dual-level retrieval, ported from graphrag-rs `lightrag::dual_retrieval`.

import Foundation

/// A single retrieval hit at one abstraction level.
public struct LightRAGResult: Sendable, Equatable {
    public var id: String
    public var content: String
    public var score: Float
    public var entities: [String]
    public var sourceChunks: [String]

    public init(
        id: String, content: String, score: Float,
        entities: [String] = [], sourceChunks: [String] = []
    ) {
        self.id = id
        self.content = content
        self.score = score
        self.entities = entities
        self.sourceChunks = sourceChunks
    }
}

/// A store that can be searched semantically for one retrieval level.
public protocol SemanticSearcher: Sendable {
    func search(_ query: String, topK: Int) async throws -> [LightRAGResult]
}

/// How high- and low-level result sets are combined.
public enum MergeStrategy: Sendable, Equatable {
    /// Alternate one high, one low, … (default).
    case interleave
    /// All high-level first, then low-level.
    case highFirst
    /// All low-level first, then high-level.
    case lowFirst
    /// Sort by level-weighted score, descending.
    case weighted
}

/// Configuration for `DualLevelRetriever`.
public struct DualRetrievalConfig: Sendable {
    public var highLevelWeight: Float
    public var lowLevelWeight: Float
    public var mergeStrategy: MergeStrategy

    public init(
        highLevelWeight: Float = 0.6,
        lowLevelWeight: Float = 0.4,
        mergeStrategy: MergeStrategy = .interleave
    ) {
        self.highLevelWeight = highLevelWeight
        self.lowLevelWeight = lowLevelWeight
        self.mergeStrategy = mergeStrategy
    }
}

/// The output of a dual-level retrieval.
public struct DualRetrievalResults: Sendable {
    public var highLevelChunks: [LightRAGResult]
    public var lowLevelChunks: [LightRAGResult]
    public var mergedChunks: [LightRAGResult]
    public var keywords: DualLevelKeywords

    public init(
        highLevelChunks: [LightRAGResult], lowLevelChunks: [LightRAGResult],
        mergedChunks: [LightRAGResult], keywords: DualLevelKeywords
    ) {
        self.highLevelChunks = highLevelChunks
        self.lowLevelChunks = lowLevelChunks
        self.mergedChunks = mergedChunks
        self.keywords = keywords
    }
}

/// Runs LightRAG dual-level retrieval: extract high-/low-level keywords, search
/// each level's store, and merge the results.
public struct DualLevelRetriever: Sendable {
    public var keywordExtractor: KeywordExtractor
    public var highLevelStore: any SemanticSearcher
    public var lowLevelStore: any SemanticSearcher
    public var config: DualRetrievalConfig

    public init(
        keywordExtractor: KeywordExtractor,
        highLevelStore: any SemanticSearcher,
        lowLevelStore: any SemanticSearcher,
        config: DualRetrievalConfig = DualRetrievalConfig()
    ) {
        self.keywordExtractor = keywordExtractor
        self.highLevelStore = highLevelStore
        self.lowLevelStore = lowLevelStore
        self.config = config
    }

    public func retrieve(_ query: String, topK: Int = 10) async throws -> DualRetrievalResults {
        let keywords = await keywordExtractor.extract(query)

        // Each level searches with its joined keywords as a single query.
        let highQuery = keywords.highLevel.joined(separator: " ")
        let lowQuery = keywords.lowLevel.joined(separator: " ")
        let high = highQuery.isEmpty ? [] : try await highLevelStore.search(highQuery, topK: topK)
        let low = lowQuery.isEmpty ? [] : try await lowLevelStore.search(lowQuery, topK: topK)

        let merged = merge(high: high, low: low, topK: topK)
        return DualRetrievalResults(
            highLevelChunks: high, lowLevelChunks: low, mergedChunks: merged, keywords: keywords)
    }

    // MARK: - Merge

    private func merge(high: [LightRAGResult], low: [LightRAGResult], topK: Int) -> [LightRAGResult] {
        guard topK > 0 else { return [] }
        var seen: Set<String> = []
        var merged: [LightRAGResult] = []

        func take(_ result: LightRAGResult) {
            guard merged.count < topK, seen.insert(result.id).inserted else { return }
            merged.append(result)
        }

        switch config.mergeStrategy {
        case .interleave:
            var i = 0
            while merged.count < topK && (i < high.count || i < low.count) {
                if i < high.count { take(high[i]) }
                if i < low.count { take(low[i]) }
                i += 1
            }
        case .highFirst:
            for r in high { take(r) }
            for r in low { take(r) }
        case .lowFirst:
            for r in low { take(r) }
            for r in high { take(r) }
        case .weighted:
            let weighted =
                high.map { ($0, $0.score * config.highLevelWeight) }
                + low.map { ($0, $0.score * config.lowLevelWeight) }
            for (result, _) in weighted.sorted(by: { lhs, rhs in
                if lhs.1 == rhs.1 { return lhs.0.id < rhs.0.id }
                return lhs.1 > rhs.1
            }) {
                take(result)
            }
        }
        return merged
    }
}
