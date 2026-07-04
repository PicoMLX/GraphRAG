// Identifiers.swift
// Strongly-typed identifier wrappers, ported from graphrag-rs `core::DocumentId`,
// `core::EntityId` and `core::ChunkId`.

/// Stable identifier for a `Document`.
public struct DocumentID: Hashable, Codable, Sendable, CustomStringConvertible,
    ExpressibleByStringLiteral
{
    public var raw: String

    public init(_ raw: String) { self.raw = raw }
    public init(stringLiteral value: String) { self.raw = value }

    public var description: String { raw }
}

/// Stable identifier for an `Entity`.
public struct EntityID: Hashable, Codable, Sendable, CustomStringConvertible,
    ExpressibleByStringLiteral
{
    public var raw: String

    public init(_ raw: String) { self.raw = raw }
    public init(stringLiteral value: String) { self.raw = value }

    public var description: String { raw }
}

/// Stable identifier for a `TextChunk`.
public struct ChunkID: Hashable, Codable, Sendable, CustomStringConvertible,
    ExpressibleByStringLiteral
{
    public var raw: String

    public init(_ raw: String) { self.raw = raw }
    public init(stringLiteral value: String) { self.raw = value }

    public var description: String { raw }
}
