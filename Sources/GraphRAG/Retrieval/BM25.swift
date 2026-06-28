// BM25.swift
// Ported from graphrag-rs `retrieval::bm25`.

import Foundation

/// A single BM25 hit.
public struct BM25Result: Sendable, Equatable {
    public var id: String
    public var score: Float
    public var content: String

    public init(id: String, score: Float, content: String) {
        self.id = id
        self.score = score
        self.content = content
    }
}

/// Okapi BM25 keyword retriever over an in-memory document collection.
///
/// Matches the Rust implementation: term frequency is normalized by document
/// length, IDF is `log(N / df) + 1`, with `k1 = 1.2` and `b = 0.75`.
public struct BM25Retriever: Sendable {
    public let k1: Float
    public let b: Float

    private struct Entry {
        var content: String
        var length: Int
        var termCounts: [String: Int]
    }

    private var entries: [String: Entry] = [:]
    private var order: [String] = []
    private var documentFrequency: [String: Int] = [:]
    private var totalLength: Int = 0

    public init(k1: Float = 1.2, b: Float = 0.75) {
        self.k1 = k1
        self.b = b
    }

    public var documentCount: Int { entries.count }
    public var termCount: Int { documentFrequency.count }
    public var averageDocumentLength: Float {
        entries.isEmpty ? 0 : Float(totalLength) / Float(entries.count)
    }

    /// Index a document under `id` with the given `content`.
    public mutating func index(id: String, content: String) {
        if entries[id] != nil { remove(id: id) }

        let tokens = BM25Retriever.tokenize(content)
        var counts: [String: Int] = [:]
        for token in tokens { counts[token, default: 0] += 1 }

        for term in counts.keys { documentFrequency[term, default: 0] += 1 }

        let entry = Entry(content: content, length: tokens.count, termCounts: counts)
        entries[id] = entry
        order.append(id)
        totalLength += tokens.count
    }

    /// Remove a previously indexed document.
    @discardableResult
    public mutating func remove(id: String) -> Bool {
        guard let entry = entries.removeValue(forKey: id) else { return false }
        order.removeAll { $0 == id }
        totalLength -= entry.length
        for term in entry.termCounts.keys {
            if let df = documentFrequency[term] {
                if df <= 1 { documentFrequency.removeValue(forKey: term) }
                else { documentFrequency[term] = df - 1 }
            }
        }
        return true
    }

    public mutating func clear() {
        entries.removeAll()
        order.removeAll()
        documentFrequency.removeAll()
        totalLength = 0
    }

    public func content(for id: String) -> String? { entries[id]?.content }

    /// Score and rank documents against `query`, returning the top `limit`.
    public func search(_ query: String, limit: Int) -> [BM25Result] {
        guard !entries.isEmpty, limit > 0 else { return [] }
        let queryTerms = Set(BM25Retriever.tokenize(query))
        guard !queryTerms.isEmpty else { return [] }

        let n = Float(entries.count)
        let avgdl = averageDocumentLength

        var results: [BM25Result] = []
        for id in order {
            guard let entry = entries[id] else { continue }
            var score: Float = 0
            for term in queryTerms {
                guard let rawCount = entry.termCounts[term], rawCount > 0 else { continue }
                let df = Float(documentFrequency[term] ?? 1)
                let idf = log(n / df) + 1.0
                let tf = Float(rawCount) / Float(max(entry.length, 1))
                let denom = tf + k1 * (1 - b + b * (Float(entry.length) / max(avgdl, 1)))
                score += idf * (tf * (k1 + 1)) / max(denom, 0.0001)
            }
            if score > 0 {
                results.append(BM25Result(id: id, score: score, content: entry.content))
            }
        }

        results.sort { lhs, rhs in
            if lhs.score == rhs.score { return lhs.id < rhs.id }
            return lhs.score > rhs.score
        }
        return Array(results.prefix(limit))
    }

    // MARK: - Tokenization

    static func tokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        for rawWord in text.split(whereSeparator: { $0.isWhitespace }) {
            var cleaned = ""
            for ch in rawWord where ch.isLetter || ch.isNumber {
                cleaned.append(contentsOf: ch.lowercased())
            }
            if cleaned.count <= 2 { continue }
            if TfIdfKeywordExtractor.defaultStopwords.contains(cleaned) { continue }
            tokens.append(cleaned)
        }
        return tokens
    }
}
