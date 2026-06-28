// VectorStore.swift
// Ported from graphrag-rs `storage` in-memory vector store.

import Foundation

/// Cosine similarity between two equal-length vectors. Returns 0 if either is
/// zero-length or dimensions mismatch.
public func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count, !a.isEmpty else { return 0 }
    var dot: Float = 0
    var normA: Float = 0
    var normB: Float = 0
    for i in 0..<a.count {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    let denom = (normA.squareRoot()) * (normB.squareRoot())
    return denom > 0 ? dot / denom : 0
}

/// A brute-force, cosine-similarity in-memory vector store.
public struct InMemoryVectorStore: Sendable {
    private var vectors: [String: [Float]] = [:]
    private var order: [String] = []

    public init() {}

    public var count: Int { vectors.count }
    public var isEmpty: Bool { vectors.isEmpty }
    public var ids: [String] { order }
    public var dimension: Int? { order.first.flatMap { vectors[$0]?.count } }

    public func contains(_ id: String) -> Bool { vectors[id] != nil }
    public func embedding(for id: String) -> [Float]? { vectors[id] }

    /// Insert or replace a vector.
    public mutating func add(id: String, vector: [Float]) {
        if vectors[id] == nil { order.append(id) }
        vectors[id] = vector
    }

    public mutating func addBatch(_ items: [(id: String, vector: [Float])]) {
        for item in items { add(id: item.id, vector: item.vector) }
    }

    @discardableResult
    public mutating func remove(id: String) -> Bool {
        guard vectors.removeValue(forKey: id) != nil else { return false }
        order.removeAll { $0 == id }
        return true
    }

    public mutating func clear() {
        vectors.removeAll()
        order.removeAll()
    }

    /// Top-`k` ids by descending cosine similarity to `query`.
    public func search(_ query: [Float], k: Int) -> [(id: String, score: Float)] {
        guard !vectors.isEmpty, k > 0 else { return [] }
        var scored: [(id: String, score: Float)] = []
        scored.reserveCapacity(order.count)
        for id in order {
            guard let v = vectors[id] else { continue }
            scored.append((id, cosineSimilarity(query, v)))
        }
        scored.sort { lhs, rhs in
            if lhs.score == rhs.score { return lhs.id < rhs.id }
            return lhs.score > rhs.score
        }
        return Array(scored.prefix(k))
    }

    /// Like `search`, but discards results below `threshold`.
    public func search(_ query: [Float], k: Int, threshold: Float) -> [(id: String, score: Float)] {
        search(query, k: k).filter { $0.score >= threshold }
    }

    /// All vectors whose similarity to `query` is at least `threshold`.
    public func findSimilar(_ query: [Float], threshold: Float) -> [(id: String, score: Float)] {
        search(query, k: order.count, threshold: threshold)
    }
}
