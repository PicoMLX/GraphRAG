// HashEmbedder.swift
// Offline, deterministic embedding backend (the default in graphrag-rs when no
// neural/remote provider is configured).
//
// Each token is hashed (FNV-1a, stable across runs) into a bucket with a signed
// contribution; the accumulated vector is L2-normalized. Texts sharing tokens
// land near each other under cosine similarity, which is enough to drive the
// retrieval pipeline without any model download or network call.

import Foundation

public struct HashEmbedder: EmbeddingModel {
    public let dimension: Int

    public init(dimension: Int = 384) {
        self.dimension = max(1, dimension)
    }

    public func isAvailable() async -> Bool { true }

    public func embed(_ text: String) async throws -> [Float] {
        embedSync(text)
    }

    /// Synchronous variant (the hashing is pure and cheap).
    public func embedSync(_ text: String) -> [Float] {
        var vector = [Float](repeating: 0, count: dimension)
        let tokens = tokenize(text)
        guard !tokens.isEmpty else { return vector }

        for token in tokens {
            let hash = HashEmbedder.fnv1a(token)
            let bucket = Int(hash % UInt64(dimension))
            let sign: Float = (hash & 0x1) == 0 ? 1 : -1
            vector[bucket] += sign
        }

        // L2 normalize.
        var norm: Float = 0
        for value in vector { norm += value * value }
        norm = norm.squareRoot()
        if norm > 0 {
            for i in 0..<dimension { vector[i] /= norm }
        }
        return vector
    }

    private func tokenize(_ text: String) -> [String] {
        // Split on any non-alphanumeric so "graph-based" hashes as the same two
        // tokens as the query "graph based" (preserves semantic overlap).
        var tokens: [String] = []
        var current = ""
        for ch in text {
            if ch.isLetter || ch.isNumber {
                current.append(contentsOf: ch.lowercased())
            } else if !current.isEmpty {
                tokens.append(current)
                current = ""
            }
        }
        if !current.isEmpty { tokens.append(current) }
        return tokens
    }

    /// 64-bit FNV-1a hash — stable across processes (unlike Swift's `Hasher`).
    static func fnv1a(_ string: String) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        let prime: UInt64 = 0x0000_0100_0000_01b3
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* prime
        }
        return hash
    }
}
