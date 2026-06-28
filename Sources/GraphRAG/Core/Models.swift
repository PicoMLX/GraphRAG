// Models.swift
// Core domain model, ported from graphrag-rs `core::mod`.

import Foundation

/// Optional metadata attached to a chunk during enrichment.
///
/// In the Rust source this is a dedicated `ChunkMetadata` struct; here it keeps
/// the most useful fields plus an open key/value bag for extensions.
public struct ChunkMetadata: Codable, Sendable, Equatable {
    /// Zero-based index of the chunk within its source document.
    public var index: Int
    /// Approximate token / word count of the chunk content.
    public var wordCount: Int
    /// Keywords extracted from the chunk, if any.
    public var keywords: [String]
    /// Arbitrary extra fields.
    public var extra: [String: String]

    public init(
        index: Int = 0,
        wordCount: Int = 0,
        keywords: [String] = [],
        extra: [String: String] = [:]
    ) {
        self.index = index
        self.wordCount = wordCount
        self.keywords = keywords
        self.extra = extra
    }
}

/// A contiguous span of a document produced by the chunking stage.
public struct TextChunk: Codable, Sendable, Identifiable, Equatable {
    public var id: ChunkID
    public var documentID: DocumentID
    public var content: String
    /// Character (grapheme) offset of the chunk start within the original
    /// document content — not a UTF-8 byte offset.
    public var startOffset: Int
    /// Character (grapheme) offset of the chunk end within the original document
    /// content — not a UTF-8 byte offset.
    public var endOffset: Int
    /// Optional dense embedding for semantic search.
    public var embedding: [Float]?
    /// Entities mentioned within this chunk.
    public var entities: [EntityID]
    public var metadata: ChunkMetadata

    public init(
        id: ChunkID,
        documentID: DocumentID,
        content: String,
        startOffset: Int,
        endOffset: Int,
        embedding: [Float]? = nil,
        entities: [EntityID] = [],
        metadata: ChunkMetadata = ChunkMetadata()
    ) {
        self.id = id
        self.documentID = documentID
        self.content = content
        self.startOffset = startOffset
        self.endOffset = endOffset
        self.embedding = embedding
        self.entities = entities
        self.metadata = metadata
    }
}

/// A source document and its derived chunks.
public struct Document: Codable, Sendable, Identifiable, Equatable {
    public var id: DocumentID
    public var title: String
    public var content: String
    public var metadata: [String: String]
    public var chunks: [TextChunk]

    public init(
        id: DocumentID,
        title: String,
        content: String,
        metadata: [String: String] = [:],
        chunks: [TextChunk] = []
    ) {
        self.id = id
        self.title = title
        self.content = content
        self.metadata = metadata
        self.chunks = chunks
    }
}

/// A single mention (occurrence) of an entity inside a chunk.
public struct EntityMention: Codable, Sendable, Equatable {
    public var chunkID: ChunkID
    public var startOffset: Int
    public var endOffset: Int
    public var confidence: Float

    public init(chunkID: ChunkID, startOffset: Int, endOffset: Int, confidence: Float) {
        self.chunkID = chunkID
        self.startOffset = startOffset
        self.endOffset = endOffset
        self.confidence = confidence
    }
}

/// A node in the knowledge graph.
public struct Entity: Codable, Sendable, Identifiable, Equatable {
    public var id: EntityID
    public var name: String
    public var entityType: String
    public var confidence: Float
    public var mentions: [EntityMention]
    public var embedding: [Float]?

    public init(
        id: EntityID,
        name: String,
        entityType: String,
        confidence: Float = 1.0,
        mentions: [EntityMention] = [],
        embedding: [Float]? = nil
    ) {
        self.id = id
        self.name = name
        self.entityType = entityType
        self.confidence = confidence
        self.mentions = mentions
        self.embedding = embedding
    }
}

/// A directed, typed edge between two entities.
public struct Relationship: Codable, Sendable, Equatable {
    public var source: EntityID
    public var target: EntityID
    public var relationType: String
    public var confidence: Float
    /// Chunks that provide evidence for this relationship.
    public var context: [ChunkID]
    public var embedding: [Float]?

    public init(
        source: EntityID,
        target: EntityID,
        relationType: String,
        confidence: Float = 1.0,
        context: [ChunkID] = [],
        embedding: [Float]? = nil
    ) {
        self.source = source
        self.target = target
        self.relationType = relationType
        self.confidence = confidence
        self.context = context
        self.embedding = embedding
    }
}
