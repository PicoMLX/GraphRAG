// LightRAGEngine.swift
// High-level LightRAG facade over a knowledge graph: detects communities,
// assembles the two dual-level stores, and answers queries.

import Foundation

/// Builds and memoizes the two dual-level stores for a fixed graph snapshot, so
/// repeated queries don't re-embed the whole corpus (which, with a remote
/// embedder, would cost O(corpus) network calls per query). Reference semantics
/// via `actor` let a value-type `LightRAGEngine` share one cache across calls.
private actor LightRAGStoreCache {
    typealias Stores = (low: any SemanticSearcher, high: any SemanticSearcher)

    private let graph: KnowledgeGraph
    private let embedder: any EmbeddingModel
    private let leidenConfig: LeidenConfig
    private var communitiesCache: CommunityDetectionResult?
    /// The single in-flight (or completed) build. Stored *before* the first
    /// await so concurrent first callers await the same task instead of each
    /// starting a duplicate O(corpus) embedding pass (actor reentrancy).
    private var storesTask: Task<Stores, Error>?

    init(graph: KnowledgeGraph, embedder: any EmbeddingModel, leidenConfig: LeidenConfig) {
        self.graph = graph
        self.embedder = embedder
        self.leidenConfig = leidenConfig
    }

    func communities() -> CommunityDetectionResult {
        if let communitiesCache { return communitiesCache }
        let detected = LeidenCommunityDetector(config: leidenConfig).detect(graph)
        communitiesCache = detected
        return detected
    }

    func stores() async throws -> Stores {
        if let storesTask { return try await storesTask.value }
        let detected = communities()
        let embedder = self.embedder
        let graph = self.graph
        let task = Task { () throws -> Stores in
            async let low = LightRAG.chunkSearcher(graph: graph, embedder: embedder)
            async let high = LightRAG.communitySearcher(
                graph: graph, communities: detected, embedder: embedder)
            return (low: try await low, high: try await high)
        }
        storesTask = task
        do {
            return try await task.value
        } catch {
            // Let a later call retry rather than caching the failure forever.
            storesTask = nil
            throw error
        }
    }
}

/// A self-contained LightRAG engine over a snapshot of a `KnowledgeGraph`.
///
/// The low-level store searches document chunks (entity/detail-centric); the
/// high-level store searches per-community theme summaries derived from Leiden
/// community detection (global/relationship-centric).
public struct LightRAGEngine: Sendable {
    public let graph: KnowledgeGraph
    private let embedder: any EmbeddingModel
    private let languageModel: (any LanguageModel)?
    public let config: DualRetrievalConfig
    public let keywordConfig: KeywordExtractorConfig
    public let leidenConfig: LeidenConfig
    private let cache: LightRAGStoreCache

    public init(
        graph: KnowledgeGraph,
        embedder: any EmbeddingModel,
        languageModel: (any LanguageModel)? = nil,
        config: DualRetrievalConfig = DualRetrievalConfig(),
        keywordConfig: KeywordExtractorConfig = KeywordExtractorConfig(),
        leidenConfig: LeidenConfig = LeidenConfig()
    ) {
        self.graph = graph
        self.embedder = embedder
        self.languageModel = languageModel
        self.config = config
        self.keywordConfig = keywordConfig
        self.leidenConfig = leidenConfig
        self.cache = LightRAGStoreCache(
            graph: graph, embedder: embedder, leidenConfig: leidenConfig)
    }

    /// Detect entity communities via Leiden.
    public func detectCommunities() -> CommunityDetectionResult {
        LeidenCommunityDetector(config: leidenConfig).detect(graph)
    }

    /// Run dual-level retrieval for `query`. The two stores are built once per
    /// engine (cached across calls) rather than rebuilt each query.
    public func retrieve(_ query: String, topK: Int = 10) async throws -> DualRetrievalResults {
        let stores = try await cache.stores()
        let extractor = KeywordExtractor(model: languageModel, config: keywordConfig)
        let retriever = DualLevelRetriever(
            keywordExtractor: extractor,
            highLevelStore: stores.high,
            lowLevelStore: stores.low,
            config: config)
        return try await retriever.retrieve(query, topK: topK)
    }

    /// Answer `query` using dual-level retrieval, synthesizing with the LLM when
    /// available and falling back to an extractive summary otherwise.
    public func ask(_ query: String, topK: Int = 10) async throws -> Answer {
        let results = try await retrieve(query, topK: topK)
        guard !results.mergedChunks.isEmpty else {
            return Answer(
                text: "I don't have enough information to answer this question.", confidence: 0)
        }

        let context = results.mergedChunks.map { result in
            "[Relevance: \(String(format: "%.3f", result.score))]\n\(result.content)"
        }.joined(separator: "\n\n---\n\n")

        // Ground on the real chunk ids each hit stands for (a high-level hit
        // resolves to its member chunks), deduped in first-seen order, so
        // synthetic community ids never surface as unresolvable sources.
        var seenSource: Set<String> = []
        let sources = results.mergedChunks
            .flatMap(\.sourceChunks)
            .filter { seenSource.insert($0).inserted }
            .map { ChunkID($0) }
        let confidence = min(1.0, Float(results.mergedChunks.count) / Float(max(1, topK)))

        if let languageModel, await languageModel.isAvailable() {
            let prompt = Prompts.fill(Prompts.answerGeneration, ["context": context, "query": query])
            let raw = try await languageModel.complete(prompt)
            return Answer(
                text: GraphRAG.stripThinkingTags(raw), confidence: confidence, sources: sources)
        }

        let extractive = results.mergedChunks.prefix(3).map(\.content).joined(separator: "\n\n")
        return Answer(
            text: "Based on the retrieved context:\n\n\(extractive)",
            confidence: confidence, sources: sources)
    }
}
