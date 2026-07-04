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
    private let searchOptions: LightRAGSearchOptions
    private var communitiesCache: CommunityDetectionResult?
    /// The single in-flight (or completed) build. Stored *before* the first
    /// await so concurrent first callers await the same task instead of each
    /// starting a duplicate O(corpus) embedding pass (actor reentrancy).
    private var storesTask: Task<Stores, Error>?

    init(
        graph: KnowledgeGraph, embedder: any EmbeddingModel, leidenConfig: LeidenConfig,
        searchOptions: LightRAGSearchOptions
    ) {
        self.graph = graph
        self.embedder = embedder
        self.leidenConfig = leidenConfig
        self.searchOptions = searchOptions
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
        let options = self.searchOptions
        let task = Task { () throws -> Stores in
            async let low = LightRAG.chunkSearcher(
                graph: graph, embedder: embedder, options: options)
            async let high = LightRAG.communitySearcher(
                graph: graph, communities: detected, embedder: embedder, options: options)
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
    /// Store-level retrieval settings (semantic threshold, keyword toggle),
    /// populated from the owning GraphRAG's `Config` when created via `lightRAG()`.
    public let searchOptions: LightRAGSearchOptions
    /// Default result cap used when `retrieve`/`ask` are called without an explicit
    /// `topK`; set from `Config.topKResults` via `lightRAG()`.
    public let defaultTopK: Int
    private let cache: LightRAGStoreCache

    public init(
        graph: KnowledgeGraph,
        embedder: any EmbeddingModel,
        languageModel: (any LanguageModel)? = nil,
        config: DualRetrievalConfig = DualRetrievalConfig(),
        keywordConfig: KeywordExtractorConfig = KeywordExtractorConfig(),
        leidenConfig: LeidenConfig = LeidenConfig(),
        searchOptions: LightRAGSearchOptions = LightRAGSearchOptions(),
        defaultTopK: Int = 10
    ) {
        self.graph = graph
        self.embedder = embedder
        self.languageModel = languageModel
        self.config = config
        self.keywordConfig = keywordConfig
        self.leidenConfig = leidenConfig
        self.searchOptions = searchOptions
        self.defaultTopK = defaultTopK
        self.cache = LightRAGStoreCache(
            graph: graph, embedder: embedder, leidenConfig: leidenConfig,
            searchOptions: searchOptions)
    }

    /// Detect entity communities via Leiden.
    public func detectCommunities() -> CommunityDetectionResult {
        LeidenCommunityDetector(config: leidenConfig).detect(graph)
    }

    /// Run dual-level retrieval for `query`. The two stores are built once per
    /// engine (cached across calls) rather than rebuilt each query.
    ///
    /// `topK` defaults to `defaultTopK` when omitted — which `GraphRAG.lightRAG()`
    /// sets from `Config.topKResults`, so the LightRAG path honors the same result
    /// cap as `GraphRAG.ask`/`search` on the configured instance.
    public func retrieve(_ query: String, topK: Int? = nil) async throws -> DualRetrievalResults {
        let effectiveTopK = topK ?? defaultTopK
        // Requests that can't produce hits — a nonpositive topK, or a keyword
        // budget of 0 (which forces empty high/low queries) — return before
        // building/embedding the stores, so a no-op never triggers corpus-wide
        // embedding work. (maxKeywords <= 0 is checked here directly, so no LLM
        // keyword call is made either.)
        guard effectiveTopK > 0, keywordConfig.maxKeywords > 0 else {
            return DualRetrievalResults(
                highLevelChunks: [], lowLevelChunks: [], mergedChunks: [],
                keywords: DualLevelKeywords())
        }
        let stores = try await cache.stores()
        let extractor = KeywordExtractor(model: languageModel, config: keywordConfig)
        let retriever = DualLevelRetriever(
            keywordExtractor: extractor,
            highLevelStore: stores.high,
            lowLevelStore: stores.low,
            config: config)
        return try await retriever.retrieve(query, topK: effectiveTopK)
    }

    /// Answer `query` using dual-level retrieval, synthesizing with the LLM when
    /// available and falling back to an extractive summary otherwise. `topK`
    /// defaults to `defaultTopK` (see `retrieve`).
    public func ask(_ query: String, topK: Int? = nil) async throws -> Answer {
        let effectiveTopK = topK ?? defaultTopK
        let results = try await retrieve(query, topK: effectiveTopK)
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
        let confidence = min(1.0, Float(results.mergedChunks.count) / Float(max(1, effectiveTopK)))

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
