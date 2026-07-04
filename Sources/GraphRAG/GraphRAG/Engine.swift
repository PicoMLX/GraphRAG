// Engine.swift
// Ported from graphrag-rs `graphrag::mod` / `build` / `ask`.
//
// `GraphRAG` is the high-level orchestrator. It is an `actor` so its mutable
// graph/index state is safe to share across tasks. Pluggable backends (embedder,
// optional LLM, entity extractor) are injected as existentials.

import Foundation

public actor GraphRAG {
    public let config: Config

    private var graph: KnowledgeGraph
    private let embedder: any EmbeddingModel
    private let languageModel: (any LanguageModel)?
    private let extractor: any EntityExtracting
    private let textProcessor: TextProcessor
    private var retriever: HybridRetriever
    private var isBuilt: Bool = false
    private var isBuilding: Bool = false
    /// Bumped on every ingestion so a `build()` can detect documents added while
    /// it was suspended at an `await` (actors are reentrant).
    private var ingestionVersion: Int = 0

    /// Designated initializer.
    public init(
        config: Config = .default,
        embedder: (any EmbeddingModel)? = nil,
        languageModel: (any LanguageModel)? = nil,
        extractor: (any EntityExtracting)? = nil
    ) throws {
        self.config = config
        self.graph = KnowledgeGraph()
        self.embedder = embedder ?? GraphRAG.defaultEmbedder(for: config)
        self.languageModel = languageModel
        self.extractor = extractor ?? PatternEntityExtractor(minConfidence: config.entity.minConfidence)
        self.textProcessor = try TextProcessor(
            chunkSize: config.chunkSize, chunkOverlap: config.chunkOverlap)
        self.retriever = HybridRetriever(
            config: HybridConfig(maxCandidates: max(100, config.topKResults * 10)))
    }

    /// Pick the default embedder honoring `config.embedding.backend` when no
    /// embedder was injected.
    private static func defaultEmbedder(for config: Config) -> any EmbeddingModel {
        if config.embedding.backend.lowercased() == "ollama" {
            return OllamaEmbedder(
                config: OllamaConfig(embeddingDimension: config.embedding.dimension))
        }
        return HashEmbedder(dimension: config.embedding.dimension)
    }

    // MARK: - Ingestion

    /// Add raw text as a new document (auto-titled, UUID id) and chunk it.
    @discardableResult
    public func addDocument(text: String, title: String? = nil) -> DocumentID {
        let id = DocumentID(UUID().uuidString)
        let document = Document(
            id: id, title: title ?? "Document \(graph.documentCount + 1)", content: text)
        addDocument(document)
        return id
    }

    /// Add a pre-built document, chunking it if it has no chunks yet. Replacing a
    /// document with the same id drops the previous version's chunks first, so
    /// stale text can't linger in the index.
    public func addDocument(_ document: Document) {
        var doc = document
        if doc.chunks.isEmpty {
            doc.chunks = textProcessor.chunk(doc)
        }
        graph.removeChunks(forDocument: doc.id)
        graph.addDocument(doc)
        for chunk in doc.chunks { graph.addChunk(chunk) }
        isBuilt = false
        ingestionVersion += 1
    }

    // MARK: - Build

    /// Run the full indexing pipeline: extract entities/relationships, embed
    /// chunks, and build the retrieval index.
    public func build() async throws {
        guard graph.documentCount > 0 else { throw GraphRAGError.noDocuments }
        // Actors are reentrant at `await`, so refuse overlapping builds.
        guard !isBuilding else {
            throw GraphRAGError.validation(message: "A build is already in progress")
        }
        isBuilding = true
        // Any failure below leaves the system unbuilt: ask() must require a fresh,
        // successful build rather than querying half-rebuilt state.
        isBuilt = false
        defer { isBuilding = false }

        let startVersion = ingestionVersion
        graph.clearEntitiesAndRelationships()

        // Operate on a fixed snapshot of chunk ids so documents ingested mid-build
        // (which bump ingestionVersion) don't get half-processed this round.
        let chunkIDs = graph.chunks.map(\.id)

        // Stage 1: entity & relationship extraction per chunk.
        for id in chunkIDs {
            guard let chunk = graph.chunk(id) else { continue }
            let extracted = try await extractor.extract(from: chunk)
            // Apply the configured confidence threshold uniformly — injected/LLM
            // extractors don't self-filter the way the pattern extractor does.
            var entities = extracted.entities.filter {
                $0.confidence >= config.entity.minConfidence
            }
            let relationships = extracted.relationships

            // Honor the per-chunk entity cap, keeping the highest-confidence ones.
            if config.maxEntitiesPerChunk > 0, entities.count > config.maxEntitiesPerChunk {
                entities = Array(
                    entities.sorted { $0.confidence > $1.confidence }
                        .prefix(config.maxEntitiesPerChunk))
            }

            let keptIDs = Set(entities.map(\.id))
            for entity in entities { graph.addEntity(entity) }
            // `extractRelationships == false` gates insertion here. We don't try
            // to also suppress the extractor's own relationship work: the
            // EntityExtracting protocol returns entities and relationships from a
            // single call, so skipping the LLM's relationship prompting would
            // require a protocol/prompt change. Deferred deliberately.
            if config.entity.extractRelationships {
                // Scope to entities that survived THIS chunk's cap — not global
                // graph state — so the cap can't be defeated by an id that
                // already exists from an earlier chunk.
                for relationship in relationships
                where keptIDs.contains(relationship.source) && keptIDs.contains(relationship.target) {
                    graph.addRelationship(relationship)
                }
            }

            // Only annotate the chunk if it wasn't replaced during the await
            // (content still matches what we extracted from). A replacement bumps
            // ingestionVersion, so the build is already marked unbuilt and will
            // redo this next round rather than tagging new text with stale ids.
            if var current = graph.chunk(id), current.content == chunk.content {
                current.entities = entities.map(\.id)
                graph.addChunk(current)
            }
        }

        // Stage 2: embed chunks — skipped entirely for keyword-only retrieval,
        // which never uses embeddings (avoids embedder latency/failure, e.g. a
        // remote Ollama embedder, when only BM25 is used).
        if config.approach.lowercased() != "keyword" {
            for id in chunkIDs {
                guard let chunk = graph.chunk(id) else { continue }
                let embedding = try await embedder.embed(chunk.content)
                // Skip if the chunk was replaced during the embedding await
                // (content changed), so we never attach an old-content embedding.
                if var current = graph.chunk(id), current.content == chunk.content {
                    current.embedding = embedding
                    graph.addChunk(current)
                }
            }
        }

        // Stage 3: build the hybrid retrieval index (index(graph:) clears first).
        retriever.index(graph: graph)

        // Only declare success if no new documents arrived during the build;
        // otherwise the index is already stale and a rebuild is required.
        isBuilt = (ingestionVersion == startVersion)
    }

    // MARK: - Query

    /// Answer a natural-language question over the indexed corpus.
    public func ask(_ query: String) async throws -> Answer {
        guard isBuilt else { throw GraphRAGError.notInitialized }

        let results = try await runRetrieval(query, limit: config.topKResults)

        guard !results.isEmpty else {
            return Answer(
                text: "I don't have enough information to answer this question.",
                confidence: 0)
        }

        let context = assembleContext(results)
        let sources = results.map { ChunkID($0.id) }
        let confidence = min(1.0, Float(results.count) / Float(max(1, config.topKResults)))

        // If an LLM is configured, synthesize a natural-language answer.
        if let languageModel, await languageModel.isAvailable() {
            let prompt = Prompts.fill(
                Prompts.answerGeneration, ["context": context, "query": query])
            let raw = try await languageModel.complete(prompt)
            return Answer(
                text: GraphRAG.stripThinkingTags(raw), confidence: confidence, sources: sources)
        }

        // Otherwise return an extractive summary of the top chunks.
        let extractive = results.prefix(3).map(\.content).joined(separator: "\n\n")
        return Answer(
            text: "Based on the retrieved context:\n\n\(extractive)",
            confidence: confidence, sources: sources)
    }

    /// Hybrid search without answer synthesis.
    public func search(_ query: String, limit: Int? = nil) async throws -> [HybridSearchResult] {
        guard isBuilt else { throw GraphRAGError.notInitialized }
        return try await runRetrieval(query, limit: limit ?? config.topKResults)
    }

    /// Run retrieval honoring the configured `approach` (hybrid / keyword /
    /// semantic) and the top-level `similarityThreshold`.
    private func runRetrieval(_ query: String, limit: Int) async throws -> [HybridSearchResult] {
        let approach = config.approach.lowercased()
        let includeKeyword = approach != "semantic"
        let includeSemantic = approach != "keyword"
        let queryEmbedding = includeSemantic ? try await embedder.embed(query) : nil
        return retriever.search(
            query: query,
            queryEmbedding: queryEmbedding,
            limit: limit,
            semanticThreshold: config.similarityThreshold,
            includeKeyword: includeKeyword,
            includeSemantic: includeSemantic)
    }

    // MARK: - Introspection

    public func stats() -> Stats {
        Stats(
            documentCount: graph.documentCount,
            chunkCount: graph.chunkCount,
            entityCount: graph.entityCount,
            relationshipCount: graph.relationshipCount)
    }

    /// Direct access to the underlying knowledge graph (a value-type snapshot).
    public func knowledgeGraph() -> KnowledgeGraph { graph }

    /// Detect entity communities in the current graph via Leiden. Requires a
    /// successful `build()` first, so communities reflect extracted entities and
    /// not a raw or stale graph (mirrors the `ask`/`search` guard).
    public func detectCommunities(config: LeidenConfig = LeidenConfig()) throws
        -> CommunityDetectionResult
    {
        guard isBuilt else { throw GraphRAGError.notInitialized }
        return LeidenCommunityDetector(config: config).detect(graph)
    }

    /// A LightRAG dual-level retrieval engine over the current graph snapshot,
    /// wired to this instance's embedder and (optional) language model. Requires
    /// a successful `build()` first, so the engine can't hand out answers from a
    /// raw or partially rebuilt graph (mirrors the `ask`/`search` guard).
    public func lightRAG(
        config: DualRetrievalConfig = DualRetrievalConfig(),
        keywordConfig: KeywordExtractorConfig = KeywordExtractorConfig(),
        leidenConfig: LeidenConfig = LeidenConfig()
    ) throws -> LightRAGEngine {
        guard isBuilt else { throw GraphRAGError.notInitialized }
        // LightRAG does dual-level *semantic* retrieval, which needs chunk
        // embeddings. A keyword-only build (`approach == "keyword"`) leaves those
        // nil on purpose, so reject the mismatch explicitly rather than forcing
        // corpus-wide re-embedding — which would also throw outright if this
        // instance's embedder is remote/unavailable.
        guard self.config.approach.lowercased() != "keyword" else {
            throw GraphRAGError.validation(
                message: "LightRAG requires embeddings and is unavailable with approach == \"keyword\"")
        }
        return LightRAGEngine(
            graph: graph, embedder: embedder, languageModel: languageModel,
            config: config, keywordConfig: keywordConfig, leidenConfig: leidenConfig)
    }

    /// Persist the knowledge graph to JSON.
    public func save(toJSON path: String) throws { try graph.save(toJSON: path) }

    // MARK: - Helpers

    private func assembleContext(_ results: [HybridSearchResult]) -> String {
        results.map { result in
            let score = String(format: "%.3f", result.score)
            return "[Chunk | Relevance: \(score)]\n\(result.content)"
        }.joined(separator: "\n\n---\n\n")
    }

    /// Remove `<think>...</think>` blocks emitted by some reasoning models.
    static func stripThinkingTags(_ text: String) -> String {
        var result = text
        while let open = result.range(of: "<think>"),
            let close = result.range(of: "</think>"),
            open.lowerBound < close.lowerBound
        {
            result.removeSubrange(open.lowerBound..<close.upperBound)
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
