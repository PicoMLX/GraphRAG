// Error.swift
// Ported from graphrag-rs `core::error::GraphRAGError`.

import Foundation

/// The unified error type for every fallible GraphRAG operation.
///
/// Mirrors the variants of the Rust `GraphRAGError` enum. Each case carries a
/// human-readable message (and, where relevant, structured fields) so callers
/// can pattern-match or surface a description.
public enum GraphRAGError: Error, Sendable, CustomStringConvertible {
    case config(message: String)
    case notInitialized
    case noDocuments
    case io(message: String)
    case http(message: String)
    case json(message: String)
    case textProcessing(message: String)
    case graphConstruction(message: String)
    case vectorSearch(message: String)
    case entityExtraction(message: String)
    case retrieval(message: String)
    case generation(message: String)
    case functionCall(message: String)
    case storage(message: String)
    case embedding(message: String)
    case languageModel(message: String)
    case parallel(message: String)
    case serialization(message: String)
    case validation(message: String)
    case network(message: String)
    case auth(message: String)
    case notFound(resource: String, id: String)
    case alreadyExists(resource: String, id: String)
    case timeout(operation: String, seconds: Double)
    case resourceLimit(resource: String, limit: Int)
    case dataCorruption(message: String)
    case unsupported(operation: String, reason: String)
    case rateLimit(message: String)
    case conflictResolution(message: String)
    case incrementalUpdate(message: String)

    public var description: String {
        switch self {
        case .config(let m): return "Configuration error: \(m)"
        case .notInitialized: return "GraphRAG system is not initialized"
        case .noDocuments: return "No documents have been added"
        case .io(let m): return "I/O error: \(m)"
        case .http(let m): return "HTTP error: \(m)"
        case .json(let m): return "JSON error: \(m)"
        case .textProcessing(let m): return "Text processing error: \(m)"
        case .graphConstruction(let m): return "Graph construction error: \(m)"
        case .vectorSearch(let m): return "Vector search error: \(m)"
        case .entityExtraction(let m): return "Entity extraction error: \(m)"
        case .retrieval(let m): return "Retrieval error: \(m)"
        case .generation(let m): return "Generation error: \(m)"
        case .functionCall(let m): return "Function call error: \(m)"
        case .storage(let m): return "Storage error: \(m)"
        case .embedding(let m): return "Embedding error: \(m)"
        case .languageModel(let m): return "Language model error: \(m)"
        case .parallel(let m): return "Parallel processing error: \(m)"
        case .serialization(let m): return "Serialization error: \(m)"
        case .validation(let m): return "Validation error: \(m)"
        case .network(let m): return "Network error: \(m)"
        case .auth(let m): return "Authentication error: \(m)"
        case .notFound(let resource, let id):
            return "\(resource) not found: \(id)"
        case .alreadyExists(let resource, let id):
            return "\(resource) already exists: \(id)"
        case .timeout(let operation, let seconds):
            return "Operation '\(operation)' timed out after \(seconds)s"
        case .resourceLimit(let resource, let limit):
            return "Resource limit exceeded for \(resource): \(limit)"
        case .dataCorruption(let m): return "Data corruption: \(m)"
        case .unsupported(let operation, let reason):
            return "Unsupported operation '\(operation)': \(reason)"
        case .rateLimit(let m): return "Rate limit exceeded: \(m)"
        case .conflictResolution(let m): return "Conflict resolution error: \(m)"
        case .incrementalUpdate(let m): return "Incremental update error: \(m)"
        }
    }
}

/// Convenience matching the Rust `pub type Result<T> = ...` alias.
public typealias GraphRAGResult<T> = Swift.Result<T, GraphRAGError>
