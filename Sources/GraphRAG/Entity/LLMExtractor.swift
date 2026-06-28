// LLMExtractor.swift
// Ported from graphrag-rs `entity::llm_extractor`.

import Foundation

/// LLM-driven entity & relationship extractor.
///
/// Builds the extraction prompt, calls a `LanguageModel`, and parses the JSON
/// response with the same staged fallbacks as the Rust version (direct decode →
/// fenced code block → first/last brace slice).
public struct LLMEntityExtractor<Model: LanguageModel>: EntityExtracting {
    public let model: Model
    public var entityTypes: [String]
    public var temperature: Float
    public var maxTokens: Int
    /// Extra gleaning passes to recover missed items (0 = single pass).
    public var gleaningRounds: Int

    public init(
        model: Model,
        entityTypes: [String] = Prompts.defaultEntityTypes,
        temperature: Float = 0.0,
        maxTokens: Int = 1500,
        gleaningRounds: Int = 0
    ) {
        self.model = model
        self.entityTypes = entityTypes
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.gleaningRounds = gleaningRounds
    }

    public func extract(from chunk: TextChunk) async throws
        -> (entities: [Entity], relationships: [Relationship])
    {
        let typesList = entityTypes.joined(separator: ", ")
        let prompt = Prompts.fill(
            Prompts.entityExtraction,
            ["entity_types": typesList, "input_text": chunk.content])
        let params = GenerationParams(maxTokens: maxTokens, temperature: temperature)
        let response = try await model.complete(prompt, params: params)

        var output = LLMEntityExtractor.parse(response) ?? ExtractionOutput()

        // Optional gleaning passes.
        var round = 0
        while round < gleaningRounds {
            let prevEntities = output.entities.map { "- \($0.name) (\($0.type))" }
                .joined(separator: "\n")
            let prevRelationships = output.relationships
                .map { "- \($0.source) -> \($0.target)" }.joined(separator: "\n")
            let gleanPrompt = Prompts.fill(
                Prompts.gleaningContinuation,
                [
                    "entity_types": typesList,
                    "input_text": chunk.content,
                    "previous_entities": prevEntities.isEmpty ? "(none)" : prevEntities,
                    "previous_relationships": prevRelationships.isEmpty ? "(none)" : prevRelationships,
                ])
            let gleanResponse = try await model.complete(gleanPrompt, params: params)
            if let extra = LLMEntityExtractor.parse(gleanResponse) {
                if extra.entities.isEmpty && extra.relationships.isEmpty { break }
                output.entities.append(contentsOf: extra.entities)
                output.relationships.append(contentsOf: extra.relationships)
            } else {
                break
            }
            round += 1
        }

        return convert(output, chunk: chunk)
    }

    // MARK: - Conversion

    private func convert(_ output: ExtractionOutput, chunk: TextChunk)
        -> (entities: [Entity], relationships: [Relationship])
    {
        var entities: [Entity] = []
        var idByName: [String: EntityID] = [:]
        let lowerContent = chunk.content.lowercased()

        for data in output.entities {
            let name = data.name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { continue }
            let type = data.type.isEmpty ? "CONCEPT" : data.type.uppercased()
            let id = PatternEntityExtractor.makeEntityID(type: type, name: name)

            var mentions: [EntityMention] = []
            if let range = LLMEntityExtractor.tokenBoundaryRange(
                of: name.lowercased(), in: lowerContent)
            {
                // Derive both offsets from the matched range; case folding can
                // change grapheme counts, so `start + name.count` is unreliable.
                let start = lowerContent.distance(from: lowerContent.startIndex, to: range.lowerBound)
                let end = lowerContent.distance(from: lowerContent.startIndex, to: range.upperBound)
                mentions.append(
                    EntityMention(
                        chunkID: chunk.id, startOffset: start,
                        endOffset: end, confidence: 0.9))
            }

            entities.append(
                Entity(id: id, name: name, entityType: type, confidence: 0.9, mentions: mentions))
            idByName[name.lowercased()] = id
        }

        var relationships: [Relationship] = []
        for data in output.relationships {
            let src = data.source.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            let tgt = data.target.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            guard let sourceID = idByName[src], let targetID = idByName[tgt] else { continue }
            let relType = LLMEntityExtractor.relationTypeLabel(from: data.description)
            // Clamp model-provided strength to a valid confidence; out-of-range
            // values would distort traversal filtering and PageRank weights.
            let confidence = min(max(data.strength ?? 0.7, 0), 1)
            relationships.append(
                Relationship(
                    source: sourceID, target: targetID, relationType: relType,
                    confidence: confidence, context: [chunk.id]))
        }

        return (entities, relationships)
    }

    private static func relationTypeLabel(from description: String) -> String {
        let trimmed = description.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "RELATED_TO" }
        // Use the first few words, upper-snake-cased, as a coarse relation label.
        let words = trimmed.split(whereSeparator: { $0.isWhitespace }).prefix(3)
        let label = words.map { word in
            String(word.filter { $0.isLetter || $0.isNumber }).uppercased()
        }.filter { !$0.isEmpty }.joined(separator: "_")
        return label.isEmpty ? "RELATED_TO" : label
    }

    /// First occurrence of `needle` in `haystack` that sits on token boundaries
    /// (not embedded inside a larger word), so "Ann" won't match "Annabelle".
    static func tokenBoundaryRange(of needle: String, in haystack: String) -> Range<String.Index>? {
        guard !needle.isEmpty else { return nil }
        func isWordChar(_ c: Character) -> Bool { c.isLetter || c.isNumber }
        var searchStart = haystack.startIndex
        while let range = haystack.range(of: needle, range: searchStart..<haystack.endIndex) {
            let beforeOK =
                range.lowerBound == haystack.startIndex
                || !isWordChar(haystack[haystack.index(before: range.lowerBound)])
            let afterOK =
                range.upperBound == haystack.endIndex
                || !isWordChar(haystack[range.upperBound])
            if beforeOK && afterOK { return range }
            searchStart = range.upperBound
        }
        return nil
    }

    // MARK: - Parsing

    struct ExtractionOutput: Codable {
        var entities: [EntityData] = []
        var relationships: [RelationshipData] = []

        init() {}

        enum CodingKeys: String, CodingKey { case entities, relationships }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            entities = (try? c.decode([EntityData].self, forKey: .entities)) ?? []
            relationships = (try? c.decode([RelationshipData].self, forKey: .relationships)) ?? []
        }
    }

    struct EntityData: Codable {
        var name: String
        var type: String
        var description: String?

        enum CodingKeys: String, CodingKey { case name, type, description }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            name = (try? c.decode(String.self, forKey: .name)) ?? ""
            type = (try? c.decode(String.self, forKey: .type)) ?? ""
            description = try? c.decode(String.self, forKey: .description)
        }
    }

    struct RelationshipData: Codable {
        var source: String
        var target: String
        var description: String
        var strength: Float?

        enum CodingKeys: String, CodingKey { case source, target, description, strength }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            source = (try? c.decode(String.self, forKey: .source)) ?? ""
            target = (try? c.decode(String.self, forKey: .target)) ?? ""
            description = (try? c.decode(String.self, forKey: .description)) ?? ""
            strength = try? c.decode(Float.self, forKey: .strength)
        }
    }

    /// Try to recover an `ExtractionOutput` from a possibly-noisy LLM response.
    static func parse(_ response: String) -> ExtractionOutput? {
        let decoder = JSONDecoder()

        // 1. Direct decode.
        if let data = response.data(using: .utf8),
            let parsed = try? decoder.decode(ExtractionOutput.self, from: data)
        {
            return parsed
        }
        // 2. Fenced code block.
        if let fenced = extractFencedJSON(response),
            let data = fenced.data(using: .utf8),
            let parsed = try? decoder.decode(ExtractionOutput.self, from: data)
        {
            return parsed
        }
        // 3. First '{' to last '}'.
        if let first = response.firstIndex(of: "{"),
            let last = response.lastIndex(of: "}"), first < last
        {
            let slice = String(response[first...last])
            if let data = slice.data(using: .utf8),
                let parsed = try? decoder.decode(ExtractionOutput.self, from: data)
            {
                return parsed
            }
        }
        return nil
    }

    private static func extractFencedJSON(_ text: String) -> String? {
        guard let fenceStart = text.range(of: "```") else { return nil }
        var afterFence = text[fenceStart.upperBound...]
        // Skip an optional language tag line ("json").
        if let newline = afterFence.firstIndex(of: "\n") {
            let firstLine = afterFence[afterFence.startIndex..<newline]
                .trimmingCharacters(in: .whitespaces)
            if firstLine.lowercased() == "json" || firstLine.isEmpty {
                afterFence = afterFence[afterFence.index(after: newline)...]
            }
        }
        guard let closing = afterFence.range(of: "```") else { return nil }
        return String(afterFence[afterFence.startIndex..<closing.lowerBound])
    }
}
