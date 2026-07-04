// KeywordExtraction.swift
// Ported from graphrag-rs `text::keyword_extraction` (TfIdfKeywordExtractor).

import Foundation

/// TF-IDF keyword extractor.
///
/// Maintains corpus document frequencies so IDF can be computed across a growing
/// collection. With an empty corpus every term has an assumed document frequency
/// of 1 (treated as rare), so scoring degrades gracefully to plain TF weighting.
public struct TfIdfKeywordExtractor: Sendable {
    public private(set) var documentFrequencies: [String: Int]
    public private(set) var totalDocuments: Int
    public let stopwords: Set<String>

    public init(documentFrequencies: [String: Int] = [:], totalDocuments: Int = 0) {
        self.documentFrequencies = documentFrequencies
        // Start at the true count (0 for a fresh corpus). The smoothed IDF below
        // handles an empty corpus without a phantom document.
        self.totalDocuments = max(0, totalDocuments)
        self.stopwords = TfIdfKeywordExtractor.defaultStopwords
    }

    /// Extract the top-`topK` `(term, score)` pairs, sorted by descending score.
    public func extractKeywords(_ text: String, topK: Int) -> [(term: String, score: Float)] {
        let tokens = tokenize(text)
        guard !tokens.isEmpty, topK > 0 else { return [] }

        // Term frequency (normalized by document length).
        var counts: [String: Int] = [:]
        for token in tokens { counts[token, default: 0] += 1 }
        let totalTerms = Float(tokens.count)

        var scored: [(term: String, score: Float)] = []
        scored.reserveCapacity(counts.count)
        for (term, count) in counts {
            let tf = Float(count) / totalTerms
            let idf = inverseDocumentFrequency(term)
            scored.append((term, tf * idf))
        }

        scored.sort { lhs, rhs in
            if lhs.score == rhs.score { return lhs.term < rhs.term }
            return lhs.score > rhs.score
        }
        return Array(scored.prefix(topK))
    }

    /// Extract just the top-`topK` keyword strings.
    public func extractKeywordStrings(_ text: String, topK: Int) -> [String] {
        extractKeywords(text, topK: topK).map(\.term)
    }

    /// Add a document's terms to the corpus statistics (for IDF).
    public mutating func addDocumentToCorpus(_ text: String) {
        let unique = Set(tokenize(text))
        for term in unique { documentFrequencies[term, default: 0] += 1 }
        totalDocuments += 1
    }

    public func corpusStats() -> (totalDocuments: Int, uniqueTerms: Int) {
        (totalDocuments, documentFrequencies.count)
    }

    // MARK: - Internals

    private func inverseDocumentFrequency(_ term: String) -> Float {
        let df = documentFrequencies[term] ?? 1
        // Smoothed IDF: stays strictly positive even for an empty corpus
        // (N = 1, df = 1 -> 1.0), so ranking falls back to term frequency rather
        // than collapsing every score to zero.
        let idf = log(Float(totalDocuments + 1) / Float(df + 1)) + 1.0
        return max(idf, 0.0)
    }

    /// Lowercase, keep alphanumerics/`-`/`_`, drop short / numeric / stopword tokens.
    func tokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        for rawWord in text.split(whereSeparator: { $0.isWhitespace }) {
            var cleaned = ""
            for ch in rawWord where ch.isLetter || ch.isNumber || ch == "-" || ch == "_" {
                cleaned.append(contentsOf: ch.lowercased())
            }
            if cleaned.count <= 2 { continue }
            if stopwords.contains(cleaned) { continue }
            if cleaned.allSatisfy({ $0.isNumber }) { continue }
            tokens.append(cleaned)
        }
        return tokens
    }

    public static let defaultStopwords: Set<String> = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but",
        "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
        "my", "one", "all", "would", "there", "their", "what", "so", "up", "out",
        "if", "about", "who", "get", "which", "go", "me", "when", "make", "can",
        "like", "time", "no", "just", "him", "know", "take", "people", "into",
        "year", "your", "good", "some", "could", "them", "see", "other", "than",
        "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well",
        "way", "even", "new", "want", "because", "any", "these", "give", "day",
        "most", "us", "is", "was", "are", "been", "has", "had", "were", "said", "did",
    ]
}
