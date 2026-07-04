// Builder.swift
// Ported from graphrag-rs `builder::mod` (the fluent GraphRAGBuilder).

import Foundation

/// Fluent builder for assembling a configured `GraphRAG` instance.
///
/// ```swift
/// let rag = try GraphRAGBuilder()
///     .withChunkSize(800)
///     .withTopK(5)
///     .build()
/// ```
public struct GraphRAGBuilder: Sendable {
    private var config: Config
    private var ollamaConfig: OllamaConfig
    private var useOllamaChat: Bool = false

    public init(config: Config = .default) {
        self.config = config
        self.ollamaConfig = OllamaConfig()
    }

    // MARK: - General config

    public func withOutputDir(_ dir: String) -> Self {
        var copy = self
        copy.config.outputDir = dir
        return copy
    }

    public func withChunkSize(_ size: Int) -> Self {
        var copy = self
        copy.config.chunkSize = size
        return copy
    }

    public func withChunkOverlap(_ overlap: Int) -> Self {
        var copy = self
        copy.config.chunkOverlap = overlap
        return copy
    }

    public func withTopK(_ k: Int) -> Self {
        var copy = self
        copy.config.topKResults = k
        return copy
    }

    public func withSimilarityThreshold(_ threshold: Float) -> Self {
        var copy = self
        copy.config.similarityThreshold = threshold
        return copy
    }

    public func withApproach(_ approach: String) -> Self {
        var copy = self
        copy.config.approach = approach
        return copy
    }

    public func withEmbeddingDimension(_ dimension: Int) -> Self {
        var copy = self
        copy.config.embedding.dimension = dimension
        return copy
    }

    // MARK: - Backend selection

    /// Use the offline, deterministic hash embedder (the default).
    public func withHashEmbeddings() -> Self {
        var copy = self
        copy.config.embedding.backend = "hash"
        return copy
    }

    /// Enable a local Ollama chat model (also used for LLM-based extraction).
    public func withOllama(
        host: String = "http://localhost", port: Int = 11434, chatModel: String = "llama3.2:3b"
    ) -> Self {
        var copy = self
        copy.ollamaConfig.host = host
        copy.ollamaConfig.port = port
        copy.ollamaConfig.chatModel = chatModel
        copy.useOllamaChat = true
        return copy
    }

    /// Use Ollama for embeddings instead of the hash embedder.
    public func withOllamaEmbeddings(model: String = "nomic-embed-text", dimension: Int = 1024) -> Self {
        var copy = self
        copy.ollamaConfig.embeddingModel = model
        copy.ollamaConfig.embeddingDimension = dimension
        copy.config.embedding.backend = "ollama"
        copy.config.embedding.dimension = dimension
        return copy
    }

    /// Preconfigure for a fully local Ollama setup (chat + embeddings).
    public func withLocalDefaults() -> Self {
        self.withOllama().withOllamaEmbeddings()
    }

    public func withConfig(_ config: Config) -> Self {
        var copy = self
        copy.config = config
        return copy
    }

    // MARK: - Build

    /// Construct the configured `GraphRAG` engine.
    public func build() throws -> GraphRAG {
        // The embedding backend is driven solely by `config.embedding.backend`,
        // so a later `withConfig(...)` can switch it back to hash (no sticky
        // flag). Sync the Ollama embedder's dimension from the config.
        let embedder: any EmbeddingModel
        if config.embedding.backend.lowercased() == "ollama" {
            var oc = ollamaConfig
            oc.embeddingDimension = config.embedding.dimension
            embedder = OllamaEmbedder(config: oc)
        } else {
            embedder = HashEmbedder(dimension: config.embedding.dimension)
        }

        let languageModel: (any LanguageModel)? =
            useOllamaChat ? OllamaClient(config: ollamaConfig) : nil

        let extractor: any EntityExtracting
        if useOllamaChat {
            extractor = LLMEntityExtractor(model: OllamaClient(config: ollamaConfig))
        } else {
            extractor = PatternEntityExtractor(minConfidence: config.entity.minConfidence)
        }

        return try GraphRAG(
            config: config, embedder: embedder, languageModel: languageModel, extractor: extractor)
    }
}
