// KnowledgeGraph.swift
// Ported from graphrag-rs `core::KnowledgeGraph`.
//
// The Rust version is backed by petgraph plus side indexes. This port uses a
// value-type adjacency representation: entities/relationships are stored in
// insertion order with `[ID: Int]` indexes for O(1) lookup, mirroring the
// `entity_index` HashMap and IndexMap behaviour.

import Foundation

public struct KnowledgeGraph: Sendable, Codable {
    // Entities, in insertion order.
    private var entitiesByID: [EntityID: Entity]
    private var entityOrder: [EntityID]

    // Relationships, in insertion order, with adjacency indexes into the array.
    public private(set) var relationships: [Relationship]
    private var outgoing: [EntityID: [Int]]
    private var incoming: [EntityID: [Int]]

    // Documents and chunks, in insertion order.
    private var documentsByID: [DocumentID: Document]
    private var documentOrder: [DocumentID]
    private var chunksByID: [ChunkID: TextChunk]
    private var chunkOrder: [ChunkID]

    public init() {
        entitiesByID = [:]
        entityOrder = []
        relationships = []
        outgoing = [:]
        incoming = [:]
        documentsByID = [:]
        documentOrder = []
        chunksByID = [:]
        chunkOrder = []
    }

    // MARK: - Mutation

    /// Insert an entity. If one with the same id already exists, mentions are
    /// merged and the higher confidence / any available embedding is kept.
    public mutating func addEntity(_ entity: Entity) {
        if var existing = entitiesByID[entity.id] {
            existing.mentions.append(contentsOf: entity.mentions)
            existing.confidence = max(existing.confidence, entity.confidence)
            if existing.embedding == nil { existing.embedding = entity.embedding }
            if existing.entityType.isEmpty { existing.entityType = entity.entityType }
            entitiesByID[entity.id] = existing
        } else {
            entitiesByID[entity.id] = entity
            entityOrder.append(entity.id)
        }
    }

    /// Insert a directed relationship. Duplicate (source, target, type) edges are
    /// merged: their evidence context is unioned and the max confidence kept.
    public mutating func addRelationship(_ relationship: Relationship) {
        // Ignore dangling edges: both endpoints must be nodes, otherwise
        // `neighbors(of:)`/traversals could surface an EntityID with no node.
        guard entitiesByID[relationship.source] != nil,
            entitiesByID[relationship.target] != nil
        else { return }
        // Merge duplicates.
        if let existingIndices = outgoing[relationship.source] {
            for idx in existingIndices
            where relationships[idx].target == relationship.target
                && relationships[idx].relationType == relationship.relationType
            {
                relationships[idx].confidence = max(
                    relationships[idx].confidence, relationship.confidence)
                for ctx in relationship.context where !relationships[idx].context.contains(ctx) {
                    relationships[idx].context.append(ctx)
                }
                return
            }
        }
        let index = relationships.count
        relationships.append(relationship)
        outgoing[relationship.source, default: []].append(index)
        incoming[relationship.target, default: []].append(index)
    }

    public mutating func addDocument(_ document: Document) {
        if documentsByID[document.id] == nil {
            documentOrder.append(document.id)
        } else {
            // Replacing an existing id: purge the previous version's chunks so
            // direct KnowledgeGraph callers don't retain stale chunk text.
            removeChunks(forDocument: document.id)
        }
        documentsByID[document.id] = document
    }

    public mutating func addChunk(_ chunk: TextChunk) {
        if chunksByID[chunk.id] == nil { chunkOrder.append(chunk.id) }
        chunksByID[chunk.id] = chunk
        // Keep the copy embedded in its document in sync, so
        // `document(id)?.chunks` and saved JSON reflect enrichment too.
        if var doc = documentsByID[chunk.documentID] {
            if let idx = doc.chunks.firstIndex(where: { $0.id == chunk.id }) {
                doc.chunks[idx] = chunk
            } else {
                doc.chunks.append(chunk)
            }
            documentsByID[chunk.documentID] = doc
        }
    }

    /// Remove all chunks belonging to a document (used when a document is
    /// replaced so stale chunks don't survive).
    public mutating func removeChunks(forDocument documentID: DocumentID) {
        let removed = Set(chunkOrder.filter { chunksByID[$0]?.documentID == documentID })
        guard !removed.isEmpty else { return }
        chunkOrder.removeAll { removed.contains($0) }
        for id in removed { chunksByID.removeValue(forKey: id) }
        if var doc = documentsByID[documentID] {
            doc.chunks.removeAll { removed.contains($0.id) }
            documentsByID[documentID] = doc
        }
        // Scrub evidence pointing at the removed chunks: drop entity mentions and
        // relationship-context entries that reference them, so traversal/stats and
        // saved JSON don't expose facts from a document version that's gone.
        for eid in entityOrder {
            guard var entity = entitiesByID[eid], !entity.mentions.isEmpty else { continue }
            let kept = entity.mentions.filter { !removed.contains($0.chunkID) }
            if kept.count != entity.mentions.count {
                entity.mentions = kept
                entitiesByID[eid] = entity
            }
        }
        for idx in relationships.indices where !relationships[idx].context.isEmpty {
            relationships[idx].context.removeAll { removed.contains($0) }
        }
    }

    /// Drop all entities and relationships, preserving documents and chunks.
    /// Chunk entity references are cleared too (in both `chunksByID` and the
    /// document copies) so no chunk points at an entity id that no longer exists.
    public mutating func clearEntitiesAndRelationships() {
        entitiesByID.removeAll()
        entityOrder.removeAll()
        relationships.removeAll()
        outgoing.removeAll()
        incoming.removeAll()
        for id in chunkOrder where !(chunksByID[id]?.entities.isEmpty ?? true) {
            chunksByID[id]?.entities = []
        }
        for did in documentOrder {
            guard var doc = documentsByID[did] else { continue }
            for i in doc.chunks.indices where !doc.chunks[i].entities.isEmpty {
                doc.chunks[i].entities = []
            }
            documentsByID[did] = doc
        }
    }

    // MARK: - Lookup

    public func entity(_ id: EntityID) -> Entity? { entitiesByID[id] }
    public func document(_ id: DocumentID) -> Document? { documentsByID[id] }
    public func chunk(_ id: ChunkID) -> TextChunk? { chunksByID[id] }
    public func contains(_ id: EntityID) -> Bool { entitiesByID[id] != nil }

    public var entities: [Entity] { entityOrder.compactMap { entitiesByID[$0] } }
    public var documents: [Document] { documentOrder.compactMap { documentsByID[$0] } }
    public var chunks: [TextChunk] { chunkOrder.compactMap { chunksByID[$0] } }

    public var entityCount: Int { entitiesByID.count }
    public var relationshipCount: Int { relationships.count }
    public var documentCount: Int { documentsByID.count }
    public var chunkCount: Int { chunksByID.count }

    /// Bidirectional neighbors: for every incident edge, the other endpoint and
    /// the relationship. Deduplicated per (neighbor, relationType).
    public func neighbors(of id: EntityID) -> [(neighbor: EntityID, relationship: Relationship)] {
        // Keep the highest-confidence edge per (neighbor, relationType) so a weak
        // A->B can't hide a stronger reciprocal B->A from strength-filtered
        // traversals.
        var bestIndexByKey: [String: Int] = [:]
        var order: [String] = []
        func consider(_ index: Int, neighbor: EntityID) {
            let key = "\(neighbor.raw)|\(relationships[index].relationType)"
            if let existing = bestIndexByKey[key] {
                if relationships[index].confidence > relationships[existing].confidence {
                    bestIndexByKey[key] = index
                }
            } else {
                bestIndexByKey[key] = index
                order.append(key)
            }
        }
        for idx in outgoing[id] ?? [] { consider(idx, neighbor: relationships[idx].target) }
        for idx in incoming[id] ?? [] { consider(idx, neighbor: relationships[idx].source) }
        return order.map { key in
            let rel = relationships[bestIndexByKey[key]!]
            let neighbor = rel.source == id ? rel.target : rel.source
            return (neighbor, rel)
        }
    }

    /// All relationships where `id` is the source or target.
    public func entityRelationships(_ id: EntityID) -> [Relationship] {
        var out: [Relationship] = []
        for idx in outgoing[id] ?? [] { out.append(relationships[idx]) }
        for idx in incoming[id] ?? [] { out.append(relationships[idx]) }
        return out
    }

    public func outDegree(_ id: EntityID) -> Int { (outgoing[id] ?? []).count }
    public func inDegree(_ id: EntityID) -> Int { (incoming[id] ?? []).count }
    public func degree(_ id: EntityID) -> Int { outDegree(id) + inDegree(id) }

    /// Case-insensitive substring match against entity names.
    public func findEntitiesByName(_ name: String) -> [Entity] {
        let needle = name.lowercased()
        return entities.filter { $0.name.lowercased().contains(needle) }
    }

    /// Shortest path (by hop count) between two entities via BFS, inclusive of
    /// endpoints, or nil if unreachable within `maxDepth`.
    public func findRelationshipPath(
        from source: EntityID, to target: EntityID, maxDepth: Int = 5
    ) -> [EntityID]? {
        if source == target { return [source] }
        var visited: Set<EntityID> = [source]
        var queue: [(EntityID, [EntityID])] = [(source, [source])]
        while !queue.isEmpty {
            let (current, path) = queue.removeFirst()
            if path.count > maxDepth { continue }
            for (neighbor, _) in neighbors(of: current) where !visited.contains(neighbor) {
                let newPath = path + [neighbor]
                if neighbor == target { return newPath }
                visited.insert(neighbor)
                queue.append((neighbor, newPath))
            }
        }
        return nil
    }

    public func stats() -> GraphStats {
        let n = entityCount
        let avgDegree = n > 0 ? Float(2 * relationshipCount) / Float(n) : 0
        return GraphStats(
            nodeCount: n,
            edgeCount: relationshipCount,
            averageDegree: avgDegree,
            maxDepth: diameter()
        )
    }

    /// Longest shortest-path (in hops) over the undirected graph — i.e. the
    /// graph's diameter. O(V·(V+E)); intended for occasional stats calls.
    private func diameter() -> Int {
        guard entityOrder.count > 1 else { return 0 }
        var adjacency: [EntityID: [EntityID]] = [:]
        for rel in relationships {
            adjacency[rel.source, default: []].append(rel.target)
            adjacency[rel.target, default: []].append(rel.source)
        }
        var maxDist = 0
        for start in entityOrder {
            var dist: [EntityID: Int] = [start: 0]
            var queue: [EntityID] = [start]
            var head = 0
            while head < queue.count {
                let current = queue[head]
                head += 1
                let d = dist[current]!
                if d > maxDist { maxDist = d }
                for neighbor in adjacency[current] ?? [] where dist[neighbor] == nil {
                    dist[neighbor] = d + 1
                    queue.append(neighbor)
                }
            }
        }
        return maxDist
    }

    // MARK: - Codable

    private enum CodingKeys: String, CodingKey {
        case entities, relationships, documents, chunks
    }

    public init(from decoder: Decoder) throws {
        self.init()
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decodedEntities = try container.decode([Entity].self, forKey: .entities)
        let decodedDocuments = try container.decode([Document].self, forKey: .documents)
        let decodedChunks = try container.decode([TextChunk].self, forKey: .chunks)
        let decodedRelationships = try container.decode([Relationship].self, forKey: .relationships)
        for e in decodedEntities { addEntity(e) }
        for d in decodedDocuments { addDocument(d) }
        for c in decodedChunks { addChunk(c) }
        for r in decodedRelationships { addRelationship(r) }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(entities, forKey: .entities)
        try container.encode(relationships, forKey: .relationships)
        try container.encode(documents, forKey: .documents)
        try container.encode(chunks, forKey: .chunks)
    }

    /// Serialize the graph to a JSON file.
    public func save(toJSON path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            let data = try encoder.encode(self)
            try data.write(to: URL(fileURLWithPath: path))
        } catch let error as GraphRAGError {
            throw error
        } catch {
            throw GraphRAGError.io(message: error.localizedDescription)
        }
    }

    /// Load a graph from a JSON file.
    public static func load(fromJSON path: String) throws -> KnowledgeGraph {
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            return try JSONDecoder().decode(KnowledgeGraph.self, from: data)
        } catch let error as GraphRAGError {
            throw error
        } catch {
            throw GraphRAGError.io(message: error.localizedDescription)
        }
    }
}
