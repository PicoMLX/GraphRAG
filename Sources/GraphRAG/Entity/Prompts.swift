// Prompts.swift
// Ported from graphrag-rs `entity::prompts` and the answer-generation template
// in `graphrag::ask`. Templates use `{placeholder}` markers filled by callers.

import Foundation

public enum Prompts {
    /// Default entity types requested from the LLM.
    public static let defaultEntityTypes: [String] = [
        "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "CONCEPT", "OBJECT",
    ]

    /// Single-pass entity + relationship extraction prompt.
    public static let entityExtraction = """
        -Goal-
        Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

        -Steps-
        1. Identify all entities. For each identified entity, extract the following information:
        - entity_name: Name of the entity, capitalized
        - entity_type: One of the following types: [{entity_types}]
        - entity_description: Comprehensive description of the entity's attributes and activities

        2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
        For each pair of related entities, extract the following information:
        - source_entity: name of the source entity, as identified in step 1
        - target_entity: name of the target entity, as identified in step 1
        - relationship_description: explanation as to why you think the source entity and the target entity are related to each other
        - relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity

        3. Return output in JSON format with the following structure:
        {
          "entities": [
            { "name": "entity name", "type": "entity type", "description": "entity description" }
          ],
          "relationships": [
            { "source": "source entity name", "target": "target entity name", "description": "relationship description", "strength": 0.8 }
          ]
        }

        -Real Data-
        ######################
        Entity Types: {entity_types}
        Text: {input_text}
        ######################
        Output:
        """

    /// Gleaning continuation prompt to catch entities/relationships missed on
    /// the first pass.
    public static let gleaningContinuation = """
        -Goal-
        You previously extracted entities and relationships from a text document. Review your previous extraction and the original text to identify any additional entities or relationships you may have missed in the first pass.

        -Steps-
        1. Review the entities you previously identified:
        {previous_entities}

        2. Review the relationships you previously identified:
        {previous_relationships}

        3. Carefully review the original text again and identify any entities or relationships you may have missed.

        4. Return ONLY the NEW entities and relationships you discovered in this pass, using the same JSON format:
        {
          "entities": [
            { "name": "entity name", "type": "entity type", "description": "entity description" }
          ],
          "relationships": [
            { "source": "source entity name", "target": "target entity name", "description": "relationship description", "strength": 0.8 }
          ]
        }

        If you found no additional entities or relationships, return empty arrays.

        -Real Data-
        ######################
        Entity Types: {entity_types}
        Text: {input_text}
        ######################
        Output:
        """

    /// Completion-check prompt; the model answers only YES or NO.
    public static let completionCheck = """
        Based on the text below and the entities/relationships already extracted, are there any significant entities or relationships that have been missed?

        Text:
        {input_text}

        Current Entities ({entity_count}):
        {entities_summary}

        Current Relationships ({relationship_count}):
        {relationships_summary}

        Respond with ONLY "YES" if the extraction is complete and thorough, or "NO" if there are still significant entities or relationships missing.

        Answer (YES or NO):
        """

    /// Answer-generation prompt used by `GraphRAG.ask`.
    public static let answerGeneration = """
        You are a knowledgeable assistant specialized in answering questions based on a knowledge graph.

        IMPORTANT INSTRUCTIONS:
        - Answer ONLY using information from the provided context below
        - Synthesize information from ALL context sections to give a comprehensive answer
        - Provide direct, conversational, and natural responses
        - Do NOT show your reasoning process or use <think> tags
        - If the context lacks sufficient information, clearly state: "I don't have enough information to answer this question."
        - Aim for a complete answer (3-6 sentences) that covers different aspects found across the context
        - Use a natural, helpful tone as if speaking to a person

        CONTEXT:
        {context}

        QUESTION: {query}

        ANSWER (direct response only, no reasoning):
        """

    /// Fill `{key}` placeholders in `template` with `values`.
    public static func fill(_ template: String, _ values: [String: String]) -> String {
        var result = template
        for (key, value) in values {
            result = result.replacingOccurrences(of: "{\(key)}", with: value)
        }
        return result
    }
}
