// Types.swift
// Shared supporting value types, ported from graphrag-rs `core::traits` helpers.

import Foundation

/// A single hit returned by a vector store search.
public struct SearchResult: Sendable, Equatable {
    public var id: String
    /// Distance (lower is closer) — for cosine stores this is `1 - similarity`.
    public var distance: Float
    public var metadata: [String: String]?

    public init(id: String, distance: Float, metadata: [String: String]? = nil) {
        self.id = id
        self.distance = distance
        self.metadata = metadata
    }

    /// Convenience similarity score for cosine-based stores.
    public var similarity: Float { 1.0 - distance }
}

/// Knobs passed to a `LanguageModel` completion call.
public struct GenerationParams: Sendable, Equatable {
    public var maxTokens: Int?
    public var temperature: Float?
    public var topP: Float?
    public var stopSequences: [String]?

    public init(
        maxTokens: Int? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        stopSequences: [String]? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.stopSequences = stopSequences
    }

    public static let `default` = GenerationParams()
}

/// Static description of a language model.
public struct ModelInfo: Sendable, Equatable {
    public var name: String
    public var version: String?
    public var maxContextLength: Int?
    public var supportsStreaming: Bool

    public init(
        name: String,
        version: String? = nil,
        maxContextLength: Int? = nil,
        supportsStreaming: Bool = false
    ) {
        self.name = name
        self.version = version
        self.maxContextLength = maxContextLength
        self.supportsStreaming = supportsStreaming
    }
}

/// Aggregate counts describing a graph.
public struct GraphStats: Sendable, Equatable {
    public var nodeCount: Int
    public var edgeCount: Int
    public var averageDegree: Float
    public var maxDepth: Int

    public init(nodeCount: Int, edgeCount: Int, averageDegree: Float, maxDepth: Int) {
        self.nodeCount = nodeCount
        self.edgeCount = edgeCount
        self.averageDegree = averageDegree
        self.maxDepth = maxDepth
    }
}

/// Counts produced by `GraphRAG.stats()`.
public struct Stats: Sendable, Equatable {
    public var documentCount: Int
    public var chunkCount: Int
    public var entityCount: Int
    public var relationshipCount: Int

    public init(
        documentCount: Int = 0,
        chunkCount: Int = 0,
        entityCount: Int = 0,
        relationshipCount: Int = 0
    ) {
        self.documentCount = documentCount
        self.chunkCount = chunkCount
        self.entityCount = entityCount
        self.relationshipCount = relationshipCount
    }
}

/// The result of an `ask` query.
public struct Answer: Sendable, Equatable {
    public var text: String
    /// Confidence in `[0, 1]`, when available.
    public var confidence: Float
    /// Chunk identifiers used to ground the answer.
    public var sources: [ChunkID]

    public init(text: String, confidence: Float = 0.0, sources: [ChunkID] = []) {
        self.text = text
        self.confidence = confidence
        self.sources = sources
    }
}
