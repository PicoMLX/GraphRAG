// Config.swift
// Ported from graphrag-rs `config::mod`. Defaults mirror the Rust crate.

import Foundation

public struct EmbeddingConfig: Sendable {
    public var dimension: Int
    /// "hash" (offline, deterministic) or "ollama".
    public var backend: String

    public init(dimension: Int = 384, backend: String = "hash") {
        self.dimension = dimension
        self.backend = backend
    }
}

public struct GraphConfig: Sendable {
    public var maxConnections: Int
    public var threshold: Float

    public init(maxConnections: Int = 10, threshold: Float = 0.8) {
        self.maxConnections = maxConnections
        self.threshold = threshold
    }
}

public struct TextConfig: Sendable {
    public var languages: [String]

    public init(languages: [String] = ["en"]) {
        self.languages = languages
    }
}

public struct EntityConfig: Sendable {
    public var minConfidence: Float
    public var extractRelationships: Bool

    public init(minConfidence: Float = 0.7, extractRelationships: Bool = true) {
        self.minConfidence = minConfidence
        self.extractRelationships = extractRelationships
    }
}

/// Top-level GraphRAG configuration.
public struct Config: Sendable {
    public var outputDir: String
    public var chunkSize: Int
    public var chunkOverlap: Int
    public var maxEntitiesPerChunk: Int
    public var topKResults: Int
    public var similarityThreshold: Float
    /// "semantic", "keyword", or "hybrid".
    public var approach: String

    public var embedding: EmbeddingConfig
    public var graph: GraphConfig
    public var text: TextConfig
    public var entity: EntityConfig
    public var retrieval: RetrievalConfig

    public init(
        outputDir: String = "./output",
        chunkSize: Int = 1000,
        chunkOverlap: Int = 200,
        maxEntitiesPerChunk: Int = 10,
        topKResults: Int = 10,
        similarityThreshold: Float = 0.8,
        approach: String = "hybrid",
        embedding: EmbeddingConfig = EmbeddingConfig(),
        graph: GraphConfig = GraphConfig(),
        text: TextConfig = TextConfig(),
        entity: EntityConfig = EntityConfig(),
        retrieval: RetrievalConfig = RetrievalConfig()
    ) {
        self.outputDir = outputDir
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap
        self.maxEntitiesPerChunk = maxEntitiesPerChunk
        self.topKResults = topKResults
        self.similarityThreshold = similarityThreshold
        self.approach = approach
        self.embedding = embedding
        self.graph = graph
        self.text = text
        self.entity = entity
        self.retrieval = retrieval
    }

    public static let `default` = Config()
}
