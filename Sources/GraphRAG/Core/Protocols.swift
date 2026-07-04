// Protocols.swift
// Pluggable abstractions, ported from graphrag-rs `core::traits`.
//
// The Rust crate exposes both synchronous and async variants of each trait.
// In Swift we model the async variants (the ones the pipeline actually uses)
// with `async` requirements and require `Sendable` so implementations can cross
// concurrency domains.

/// A text-generation backend (the "LLM").
public protocol LanguageModel: Sendable {
    /// Complete `prompt` with default parameters.
    func complete(_ prompt: String) async throws -> String
    /// Complete `prompt` with explicit generation parameters.
    func complete(_ prompt: String, params: GenerationParams) async throws -> String
    /// Whether the backend is reachable / configured.
    func isAvailable() async -> Bool
    /// Static model description.
    var modelInfo: ModelInfo { get }
}

extension LanguageModel {
    public func complete(_ prompt: String) async throws -> String {
        try await complete(prompt, params: .default)
    }
}

/// An embedding backend that turns text into dense vectors.
public protocol EmbeddingModel: Sendable {
    /// Embed a single string.
    func embed(_ text: String) async throws -> [Float]
    /// Embed a batch of strings (default: sequential `embed`).
    func embedBatch(_ texts: [String]) async throws -> [[Float]]
    /// Dimensionality of produced vectors.
    var dimension: Int { get }
    /// Whether the backend is ready.
    func isAvailable() async -> Bool
}

extension EmbeddingModel {
    public func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        var out: [[Float]] = []
        out.reserveCapacity(texts.count)
        for text in texts {
            out.append(try await embed(text))
        }
        return out
    }
}

/// A strategy that splits raw text into chunks.
public protocol ChunkingStrategy: Sendable {
    /// Split `text` belonging to `documentID` into ordered chunks.
    func chunk(_ text: String, documentID: DocumentID) -> [TextChunk]
}

/// Extracts entities (and optionally relationships) from text.
public protocol EntityExtracting: Sendable {
    func extract(from chunk: TextChunk) async throws -> (entities: [Entity], relationships: [Relationship])
}
