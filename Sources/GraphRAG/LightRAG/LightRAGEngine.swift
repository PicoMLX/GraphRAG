// LightRAGEngine.swift
// High-level LightRAG facade over a knowledge graph: detects communities,
// assembles the two dual-level stores, and answers queries.

import Foundation

/// A self-contained LightRAG engine over a snapshot of a `KnowledgeGraph`.
///
/// The low-level store searches document chunks (entity/detail-centric); the
/// high-level store searches per-community theme summaries derived from Leiden
/// community detection (global/relationship-centric).
public struct LightRAGEngine: Sendable {
    public let graph: KnowledgeGraph
    private let embedder: any EmbeddingModel
    private let languageModel: (any LanguageModel)?
    public var config: DualRetrievalConfig
    public var keywordConfig: KeywordExtractorConfig
    public var leidenConfig: LeidenConfig

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
    }

    /// Detect entity communities via Leiden.
    public func detectCommunities() -> CommunityDetectionResult {
        LeidenCommunityDetector(config: leidenConfig).detect(graph)
    }

    /// Run dual-level retrieval for `query`.
    public func retrieve(_ query: String, topK: Int = 10) async throws -> DualRetrievalResults {
        let communities = detectCommunities()
        // The two stores are independent — build them concurrently.
        async let lowStore = LightRAG.chunkSearcher(graph: graph, embedder: embedder)
        async let highStore = LightRAG.communitySearcher(
            graph: graph, communities: communities, embedder: embedder)
        let extractor = KeywordExtractor(model: languageModel, config: keywordConfig)
        let retriever = try await DualLevelRetriever(
            keywordExtractor: extractor,
            highLevelStore: highStore,
            lowLevelStore: lowStore,
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
        // High-level hits carry virtual "community_<id>" ids that don't exist in
        // the graph; keep only real chunk ids as grounding sources.
        let sources = results.mergedChunks
            .map(\.id)
            .filter { !$0.hasPrefix("community_") }
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
