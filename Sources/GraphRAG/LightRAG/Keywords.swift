// Keywords.swift
// Dual-level keyword extraction, ported from graphrag-rs
// `lightrag::keyword_extraction`.

import Foundation

/// Keywords extracted at two levels of abstraction.
public struct DualLevelKeywords: Sendable, Equatable, Decodable {
    /// Broad topics / concepts / themes.
    public var highLevel: [String]
    /// Specific entities / attributes / details.
    public var lowLevel: [String]

    public init(highLevel: [String] = [], lowLevel: [String] = []) {
        self.highLevel = highLevel
        self.lowLevel = lowLevel
    }

    enum CodingKeys: String, CodingKey {
        case highLevel = "high_level"
        case lowLevel = "low_level"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        highLevel = (try? c.decode([String].self, forKey: .highLevel)) ?? []
        lowLevel = (try? c.decode([String].self, forKey: .lowLevel)) ?? []
    }

    public var total: Int { highLevel.count + lowLevel.count }
    public var isEmpty: Bool { total == 0 }
}

/// Configuration for `KeywordExtractor`.
public struct KeywordExtractorConfig: Sendable {
    public var maxKeywords: Int
    public var language: String

    public init(maxKeywords: Int = 20, language: String = "English") {
        self.maxKeywords = maxKeywords
        self.language = language
    }
}

/// Extracts high-/low-level keywords from a query using an LLM, with a
/// deterministic non-LLM fallback.
public struct KeywordExtractor: Sendable {
    public let model: (any LanguageModel)?
    public var config: KeywordExtractorConfig

    public init(model: (any LanguageModel)? = nil, config: KeywordExtractorConfig = KeywordExtractorConfig()) {
        self.model = model
        self.config = config
    }

    /// Extract dual-level keywords. Never throws — any LLM/parse failure falls
    /// back to a simple query tokenization.
    public func extract(_ query: String) async -> DualLevelKeywords {
        guard let model, await model.isAvailable() else {
            return capped(KeywordExtractor.fallback(query))
        }
        let prompt = buildPrompt(query)
        do {
            let response = try await model.complete(prompt)
            if let parsed = KeywordExtractor.parse(response), !parsed.isEmpty {
                return capped(parsed)
            }
        } catch {
            // fall through to fallback
        }
        // The fallback is capped too, so `maxKeywords` holds even without an LLM.
        return capped(KeywordExtractor.fallback(query))
    }

    /// Cap the combined keyword count at `maxKeywords`, keeping high-level first.
    private func capped(_ keywords: DualLevelKeywords) -> DualLevelKeywords {
        guard keywords.total > config.maxKeywords else { return keywords }
        var high = keywords.highLevel
        var low = keywords.lowLevel
        if high.count > config.maxKeywords { high = Array(high.prefix(config.maxKeywords)) }
        let remaining = max(0, config.maxKeywords - high.count)
        low = Array(low.prefix(remaining))
        return DualLevelKeywords(highLevel: high, lowLevel: low)
    }

    func buildPrompt(_ query: String) -> String {
        """
        Extract keywords at two levels from this query: "\(query)"

        Return JSON with this exact structure:
        {
          "high_level": ["theme1", "theme2", ...],
          "low_level": ["entity1", "entity2", ...]
        }

        Rules:
        1. high_level: Broader topics, concepts, themes (abstract level)
        2. low_level: Specific entities, attributes, details (concrete level)
        3. LIMIT: Maximum \(config.maxKeywords) total keywords combined
        4. NO duplication between levels
        5. Keep keywords concise (1-3 words each)

        Example 1:
        Query: "How did Alice and Bob collaborate on the quantum computing project?"
        {
          "high_level": ["collaboration", "quantum computing", "teamwork"],
          "low_level": ["Alice", "Bob", "project"]
        }

        Example 2:
        Query: "What are the main themes in the dataset?"
        {
          "high_level": ["themes", "patterns", "overview"],
          "low_level": ["dataset"]
        }

        Language: \(config.language)

        Now extract keywords:
        """
    }

    /// Parse the JSON object between the first `{` and last `}` of the response.
    /// Thinking-tag blocks are stripped first, so a `{` inside a `<think>…</think>`
    /// preamble can't be mistaken for the start of the JSON payload.
    static func parse(_ response: String) -> DualLevelKeywords? {
        let cleaned = GraphRAG.stripThinkingTags(response)
        guard let first = cleaned.firstIndex(of: "{"),
            let last = cleaned.lastIndex(of: "}"), first < last
        else { return nil }
        let slice = String(cleaned[first...last])
        guard let data = slice.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(DualLevelKeywords.self, from: data)
    }

    /// Deterministic fallback: query words of length >= 4, deduplicated
    /// case-insensitively in first-seen order, up to 10, as low-level.
    static func fallback(_ query: String) -> DualLevelKeywords {
        var seen: Set<String> = []
        let words =
            query
            .split(whereSeparator: { $0.isWhitespace })
            .map { String($0.filter { $0.isLetter || $0.isNumber }) }
            .filter { $0.count >= 4 && seen.insert($0.lowercased()).inserted }
        return DualLevelKeywords(highLevel: [], lowLevel: Array(words.prefix(10)))
    }
}
