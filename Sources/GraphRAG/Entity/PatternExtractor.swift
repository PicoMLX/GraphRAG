// PatternExtractor.swift
// Ported from graphrag-rs pattern-based fallback extractor in `entity::mod`.
//
// This is the offline default: it finds Title-Case spans, classifies them with
// suffix/prefix/known-list heuristics, then infers typed relationships between
// co-occurring entities from surrounding context keywords.

import Foundation

/// Deterministic capitalization/heuristic entity extractor.
public struct PatternEntityExtractor: EntityExtracting {
    public var minConfidence: Float

    public init(minConfidence: Float = 0.5) {
        self.minConfidence = minConfidence
    }

    public func extract(from chunk: TextChunk) async throws
        -> (entities: [Entity], relationships: [Relationship])
    {
        let candidates = capitalizedSpans(in: chunk.content)

        var byName: [String: Entity] = [:]
        var orderedNames: [String] = []
        for candidate in candidates {
            guard let (type, confidence) = classify(candidate.text) else { continue }
            guard confidence >= minConfidence else { continue }
            let name = candidate.text
            let mention = EntityMention(
                chunkID: chunk.id,
                startOffset: candidate.start,
                endOffset: candidate.end,
                confidence: confidence)
            if var existing = byName[name] {
                existing.mentions.append(mention)
                existing.confidence = max(existing.confidence, confidence)
                byName[name] = existing
            } else {
                let entity = Entity(
                    id: PatternEntityExtractor.makeEntityID(type: type, name: name),
                    name: name,
                    entityType: type,
                    confidence: confidence,
                    mentions: [mention])
                byName[name] = entity
                orderedNames.append(name)
            }
        }

        let entities = orderedNames.compactMap { byName[$0] }
        let relationships = inferRelationships(entities: entities, chunk: chunk)
        return (entities, relationships)
    }

    /// Stable `"TYPE_normalized_name"` identifier.
    public static func makeEntityID(type: String, name: String) -> EntityID {
        let normalized = name.lowercased().map { ch -> Character in
            (ch.isLetter || ch.isNumber) ? ch : "_"
        }
        var collapsed = ""
        var lastUnderscore = false
        for ch in normalized {
            if ch == "_" {
                if !lastUnderscore { collapsed.append(ch) }
                lastUnderscore = true
            } else {
                collapsed.append(ch)
                lastUnderscore = false
            }
        }
        let trimmed = collapsed.trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        return EntityID("\(type.lowercased())_\(trimmed)")
    }

    // MARK: - Span detection

    private struct Span { var text: String; var start: Int; var end: Int }

    /// Maximal runs of Title-Case words (allowing a leading title like "Dr.").
    private func capitalizedSpans(in text: String) -> [Span] {
        let chars = Array(text)
        let n = chars.count
        var spans: [Span] = []
        var i = 0
        while i < n {
            if isWordStart(chars, i) && chars[i].isUppercase {
                let runStart = i
                var j = i
                // Consume consecutive capitalized words (optionally separated by a
                // single space and an optional connector like "of"/"the").
                while true {
                    // advance to end of current word
                    while j < n && !chars[j].isWhitespace { j += 1 }
                    // peek next word
                    var k = j
                    while k < n && chars[k] == " " { k += 1 }
                    if k < n && chars[k].isUppercase && k == j + 1 {
                        j = k
                        continue
                    }
                    // also allow lowercase connector ("of"/"the") between caps
                    if k < n && chars[k].isLowercase && k == j + 1 {
                        let connectorStart = k
                        var c = k
                        while c < n && !chars[c].isWhitespace { c += 1 }
                        let connector = String(chars[connectorStart..<c]).lowercased()
                        var after = c
                        while after < n && chars[after] == " " { after += 1 }
                        if (connector == "of" || connector == "the")
                            && after < n && chars[after].isUppercase
                        {
                            j = after
                            continue
                        }
                    }
                    break
                }
                // Trim leading/trailing punctuation on the index range so the
                // recorded offsets stay aligned with the original text (trimming
                // the string alone would leave `runStart` pointing at a dropped
                // leading character).
                let trimSet = CharacterSet(charactersIn: ".,;:!?\"'()")
                var spanStart = runStart
                var spanEnd = j
                while spanStart < spanEnd,
                    let scalar = chars[spanStart].unicodeScalars.first,
                    trimSet.contains(scalar)
                {
                    spanStart += 1
                }
                while spanEnd > spanStart,
                    let scalar = chars[spanEnd - 1].unicodeScalars.first,
                    trimSet.contains(scalar)
                {
                    spanEnd -= 1
                }
                if spanEnd - spanStart >= 2 {
                    let cleaned = String(chars[spanStart..<spanEnd])
                    spans.append(Span(text: cleaned, start: spanStart, end: spanEnd))
                }
                i = j
            } else {
                i += 1
            }
        }
        return spans
    }

    private func isWordStart(_ chars: [Character], _ i: Int) -> Bool {
        i == 0 || chars[i - 1].isWhitespace
    }

    // MARK: - Classification

    private func classify(_ text: String) -> (type: String, confidence: Float)? {
        let words = text.split(separator: " ").map(String.init)
        guard !words.isEmpty else { return nil }

        // Blocklist single sentence-initial words that are common/structural.
        if words.count == 1, PatternEntityExtractor.blocklist.contains(words[0].lowercased()) {
            return nil
        }

        // Organizations by suffix.
        if let last = words.last,
            PatternEntityExtractor.orgSuffixes.contains(last.lowercased())
        {
            return ("ORGANIZATION", 0.9)
        }
        // Organizations by prefix ("University of ...", etc.).
        let lower = text.lowercased()
        for prefix in PatternEntityExtractor.orgPrefixes where lower.hasPrefix(prefix) {
            return ("ORGANIZATION", 0.9)
        }
        // Known locations.
        if PatternEntityExtractor.knownLocations.contains(lower) {
            return ("LOCATION", 0.9)
        }
        // Titled persons.
        if let first = words.first,
            PatternEntityExtractor.personTitles.contains(
                first.trimmingCharacters(in: CharacterSet(charactersIn: ".")).lowercased())
        {
            return ("PERSON", 0.9)
        }
        // Multi-word Title Case -> likely a person/proper noun.
        if words.count >= 2 {
            return ("PERSON", 0.8)
        }
        // Single capitalized word -> generic concept.
        if words[0].count >= 3 {
            return ("CONCEPT", 0.6)
        }
        return nil
    }

    // MARK: - Relationship inference

    private func inferRelationships(entities: [Entity], chunk: TextChunk) -> [Relationship] {
        guard entities.count >= 2 else { return [] }
        let context = chunk.content.lowercased()
        var relationships: [Relationship] = []
        var seen: Set<String> = []

        for i in 0..<entities.count {
            for j in (i + 1)..<entities.count {
                let a = entities[i]
                let b = entities[j]
                let relType = relationType(for: a.entityType, b.entityType, context: context)
                let key = "\(a.id.raw)|\(b.id.raw)|\(relType)"
                if seen.contains(key) { continue }
                seen.insert(key)
                relationships.append(
                    Relationship(
                        source: a.id, target: b.id, relationType: relType,
                        confidence: 0.6, context: [chunk.id]))
            }
        }
        return relationships
    }

    private func relationType(for a: String, _ b: String, context: String) -> String {
        func has(_ s: String) -> Bool { context.contains(s) }
        switch (a, b) {
        case ("PERSON", "ORGANIZATION"), ("ORGANIZATION", "PERSON"):
            if has("works for") || has("employed by") { return "WORKS_FOR" }
            if has("founded") || has("ceo") { return "LEADS" }
            return "ASSOCIATED_WITH"
        case ("PERSON", "LOCATION"), ("LOCATION", "PERSON"):
            if has("born in") || has(" from ") { return "BORN_IN" }
            if has("lives in") || has("based in") { return "LOCATED_IN" }
            return "ASSOCIATED_WITH"
        case ("ORGANIZATION", "LOCATION"), ("LOCATION", "ORGANIZATION"):
            if has("headquartered") || has("based in") { return "HEADQUARTERED_IN" }
            return "LOCATED_IN"
        case ("PERSON", "PERSON"):
            if has("married") || has("spouse") { return "MARRIED_TO" }
            if has("colleague") || has("partner") { return "COLLEAGUE_OF" }
            return "KNOWS"
        default:
            return "RELATED_TO"
        }
    }

    // MARK: - Lexicons

    static let orgSuffixes: Set<String> = [
        "inc", "inc.", "corp", "corp.", "llc", "ltd", "ltd.", "company",
        "corporation", "group", "solutions", "technologies",
    ]
    static let orgPrefixes: [String] = ["university of", "institute of", "department of"]
    static let knownLocations: Set<String> = [
        "united states", "new york", "california", "london", "paris", "tokyo",
        "berlin", "washington", "boston", "chicago",
    ]
    static let personTitles: Set<String> = ["dr", "prof", "mr", "mrs", "ms"]
    static let blocklist: Set<String> = [
        "the", "and", "but", "or", "chapter", "section", "however", "therefore",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "this", "that", "these",
        "those", "there", "here", "when", "where", "what", "who", "why", "how",
    ]
}
