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
                    // Clause punctuation (comma/semicolon/colon) ends the run so
                    // "Alice, Bob" stays two entities. Sentence punctuation
                    // (./!/?) also ends it ("Acme. Bob") — unless the word is a
                    // known abbreviation/title like "Dr." so "Dr. Smith" merges.
                    if j > runStart, let last = chars[j - 1].unicodeScalars.first {
                        if CharacterSet(charactersIn: ",;:").contains(last) { break }
                        if CharacterSet(charactersIn: ".!?").contains(last) {
                            var ws = j - 1
                            while ws > runStart && !chars[ws - 1].isWhitespace { ws -= 1 }
                            let word = String(chars[ws..<j])
                                .trimmingCharacters(in: CharacterSet(charactersIn: ".!?"))
                                .lowercased()
                            // Only person titles (which grammatically precede a
                            // name) cross the period — "Dr. Smith" merges, but an
                            // org suffix like "Acme Inc. Bob" must split.
                            if !PatternEntityExtractor.personTitles.contains(word) { break }
                        }
                    }
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
        if i == 0 { return true }
        let prev = chars[i - 1]
        // Start a run after whitespace or opening punctuation, so quoted or
        // parenthesized names (e.g. "Ada Lovelace" or (Paris)) aren't skipped.
        return prev.isWhitespace || "\"'([{".contains(prev)
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
        let chars = Array(chunk.content)

        // Assign a sentence id to every character offset (incremented after
        // ./!/?), so relationships are only inferred between entities that
        // co-occur in the SAME sentence — otherwise one "works for" phrase would
        // wrongly link every person/org pair sharing a chunk. A period that ends
        // a person title ("Dr.") is an abbreviation, not a sentence boundary, so
        // "Dr. Smith works for Acme Inc." stays one sentence.
        func periodIsSentenceEnd(_ periodIndex: Int) -> Bool {
            var s = periodIndex
            while s > 0 && chars[s - 1].isLetter { s -= 1 }
            let word = String(chars[s..<periodIndex]).lowercased()
            // Person titles precede a name, so never a sentence end ("Dr. Smith").
            if PatternEntityExtractor.personTitles.contains(word) { return false }
            // Other abbreviations (Inc./Corp.) end a sentence only when the next
            // word is capitalized — "Acme Inc. was ..." stays one sentence, but
            // "... Acme Inc. Bob ..." splits.
            if PatternEntityExtractor.sentenceAbbreviations.contains(word) {
                var t = periodIndex + 1
                while t < chars.count && chars[t] == " " { t += 1 }
                return t < chars.count && chars[t].isUppercase
            }
            return true
        }
        var sentenceID = [Int](repeating: 0, count: chars.count + 1)
        var sid = 0
        for k in 0..<chars.count {
            sentenceID[k] = sid
            let c = chars[k]
            if c == "!" || c == "?" || (c == "." && periodIsSentenceEnd(k)) { sid += 1 }
        }
        sentenceID[chars.count] = sid
        func sentence(of offset: Int) -> Int { sentenceID[max(0, min(offset, chars.count))] }

        var relationships: [Relationship] = []
        var seen: Set<String> = []

        for i in 0..<entities.count {
            for j in (i + 1)..<entities.count {
                let a = entities[i]
                let b = entities[j]
                // Find a mention pair that shares a sentence; skip the pair if none.
                guard let (aOff, bOff) = sameSentenceMentions(a, b, sentence: sentence) else {
                    continue
                }
                let lo = min(aOff, bOff)
                let hi = max(aOff, bOff)
                // Proximity heuristic: skip the pair if another entity of the same
                // type as an endpoint lies between them — the connecting phrase
                // most likely belongs to that nearer pair. (Offline extractor; not
                // a full relation classifier, so this trades some recall for far
                // fewer false edges in multi-fact sentences.)
                if hasInterveningSameType(a, b, lo: lo, hi: hi, among: entities) { continue }
                // Keyword context is just the span between the two mentions, so a
                // phrase belonging to a different pair in the sentence can't leak.
                let upper = min(hi + 1, chars.count)
                let context = (lo < upper ? String(chars[lo..<upper]) : "").lowercased()

                let relType = relationType(for: a.entityType, b.entityType, context: context)
                // Orient asymmetric relations by entity role, independent of the
                // order the spans happened to appear in the text.
                let (source, target) = orient(relType, a, b)
                let key = "\(source.id.raw)|\(target.id.raw)|\(relType)"
                if seen.contains(key) { continue }
                seen.insert(key)
                relationships.append(
                    Relationship(
                        source: source.id, target: target.id, relationType: relType,
                        confidence: 0.6, context: [chunk.id]))
            }
        }
        return relationships
    }

    /// Whether an entity (other than `a`/`b`) of the same type as one endpoint
    /// has a mention strictly between offsets `lo` and `hi`.
    private func hasInterveningSameType(
        _ a: Entity, _ b: Entity, lo: Int, hi: Int, among entities: [Entity]
    ) -> Bool {
        for c in entities where c.id != a.id && c.id != b.id {
            guard c.entityType == a.entityType || c.entityType == b.entityType else { continue }
            for m in c.mentions where m.startOffset > lo && m.startOffset < hi {
                return true
            }
        }
        return false
    }

    /// First mention pair of `a` and `b` that falls in the same sentence.
    private func sameSentenceMentions(
        _ a: Entity, _ b: Entity, sentence: (Int) -> Int
    ) -> (Int, Int)? {
        for ma in a.mentions {
            for mb in b.mentions where sentence(ma.startOffset) == sentence(mb.startOffset) {
                return (ma.startOffset, mb.startOffset)
            }
        }
        return nil
    }

    /// Order (source, target) for a typed relation by the entities' roles.
    /// Symmetric relations keep their text order.
    private func orient(_ relType: String, _ a: Entity, _ b: Entity) -> (Entity, Entity) {
        func pick(_ source: String, _ target: String) -> (Entity, Entity)? {
            if a.entityType == source && b.entityType == target { return (a, b) }
            if b.entityType == source && a.entityType == target { return (b, a) }
            return nil
        }
        switch relType {
        case "WORKS_FOR", "LEADS":
            return pick("PERSON", "ORGANIZATION") ?? (a, b)
        case "BORN_IN":
            return pick("PERSON", "LOCATION") ?? (a, b)
        case "HEADQUARTERED_IN":
            return pick("ORGANIZATION", "LOCATION") ?? (a, b)
        case "LOCATED_IN":
            // Whichever endpoint is the location is the target.
            if a.entityType == "LOCATION" { return (b, a) }
            if b.entityType == "LOCATION" { return (a, b) }
            return (a, b)
        default:
            return (a, b)  // ASSOCIATED_WITH / KNOWS / MARRIED_TO / RELATED_TO ...
        }
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
    /// Words whose trailing period is an abbreviation rather than a sentence end,
    /// used only for sentence segmentation in relationship inference (so
    /// "Acme Inc. was founded by Sam Altman" stays one sentence). Entity-span
    /// splitting still uses the narrower `personTitles`.
    static let sentenceAbbreviations: Set<String> = [
        "dr", "prof", "mr", "mrs", "ms", "jr", "sr", "st",
        "inc", "corp", "ltd", "llc", "co", "etc", "vs",
    ]
    static let blocklist: Set<String> = [
        "the", "and", "but", "or", "chapter", "section", "however", "therefore",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "this", "that", "these",
        "those", "there", "here", "when", "where", "what", "who", "why", "how",
    ]
}
