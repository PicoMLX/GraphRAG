// Traversal.swift
// Ported from graphrag-rs `graph::traversal`.

import Foundation

/// Tunables that govern graph traversal.
public struct TraversalConfig: Sendable {
    public var maxDepth: Int
    public var maxPaths: Int
    public var useEdgeWeights: Bool
    public var minRelationshipStrength: Float

    public init(
        maxDepth: Int = 3,
        maxPaths: Int = 100,
        useEdgeWeights: Bool = true,
        minRelationshipStrength: Float = 0.5
    ) {
        self.maxDepth = maxDepth
        self.maxPaths = maxPaths
        self.useEdgeWeights = useEdgeWeights
        self.minRelationshipStrength = minRelationshipStrength
    }
}

/// The product of a traversal: discovered entities, the edges walked, and the
/// depth/distance of each entity from the source(s).
public struct TraversalResult: Sendable {
    public var entities: [EntityID]
    public var relationships: [Relationship]
    public var distances: [EntityID: Int]

    public init(
        entities: [EntityID] = [],
        relationships: [Relationship] = [],
        distances: [EntityID: Int] = [:]
    ) {
        self.entities = entities
        self.relationships = relationships
        self.distances = distances
    }
}

/// Breadth-/depth-first traversal of the knowledge graph with edge-strength
/// filtering.
public struct GraphTraversal: Sendable {
    public var config: TraversalConfig

    public init(config: TraversalConfig = TraversalConfig()) {
        self.config = config
    }

    private func passesFilter(_ relationship: Relationship) -> Bool {
        !config.useEdgeWeights || relationship.confidence >= config.minRelationshipStrength
    }

    /// Breadth-first search from a single source.
    public func bfs(_ graph: KnowledgeGraph, from source: EntityID) -> TraversalResult {
        multiSourceBFS(graph, from: [source])
    }

    /// Breadth-first search from multiple sources simultaneously.
    public func multiSourceBFS(_ graph: KnowledgeGraph, from sources: [EntityID]) -> TraversalResult {
        var result = TraversalResult()
        var visited: Set<EntityID> = []
        var queue: [EntityID] = []
        for source in sources where graph.contains(source) && !visited.contains(source) {
            visited.insert(source)
            result.distances[source] = 0
            result.entities.append(source)
            queue.append(source)
        }

        var head = 0
        while head < queue.count {
            let current = queue[head]
            head += 1
            let depth = result.distances[current] ?? 0
            if depth >= config.maxDepth { continue }
            for (neighbor, relationship) in graph.neighbors(of: current) {
                guard passesFilter(relationship) else { continue }
                if !visited.contains(neighbor) {
                    visited.insert(neighbor)
                    result.distances[neighbor] = depth + 1
                    result.entities.append(neighbor)
                    result.relationships.append(relationship)
                    queue.append(neighbor)
                }
            }
        }
        return result
    }

    /// Depth-first search from a single source.
    public func dfs(_ graph: KnowledgeGraph, from source: EntityID) -> TraversalResult {
        var result = TraversalResult()
        guard graph.contains(source) else { return result }
        var visited: Set<EntityID> = []
        dfsVisit(graph, current: source, depth: 0, visited: &visited, result: &result)
        return result
    }

    private func dfsVisit(
        _ graph: KnowledgeGraph,
        current: EntityID,
        depth: Int,
        visited: inout Set<EntityID>,
        result: inout TraversalResult
    ) {
        if depth > config.maxDepth || visited.contains(current) { return }
        visited.insert(current)
        result.distances[current] = depth
        result.entities.append(current)
        // Stop expanding at the depth limit so we never record an edge to a node
        // that won't itself be visited (matches BFS and the documented limit).
        guard depth < config.maxDepth else { return }
        for (neighbor, relationship) in graph.neighbors(of: current) {
            guard passesFilter(relationship) else { continue }
            if !visited.contains(neighbor) {
                result.relationships.append(relationship)
                dfsVisit(graph, current: neighbor, depth: depth + 1, visited: &visited, result: &result)
            }
        }
    }

    /// k-hop ego network expanding layer by layer around `center`.
    public func egoNetwork(_ graph: KnowledgeGraph, center: EntityID, hops: Int? = nil) -> TraversalResult {
        let k = hops ?? config.maxDepth
        var result = TraversalResult()
        guard graph.contains(center) else { return result }
        var visited: Set<EntityID> = [center]
        result.distances[center] = 0
        result.entities.append(center)
        var currentLayer = [center]
        // De-duplicate emitted edges: neighbors of adjacent layers can revisit
        // the same edge, which would otherwise overcount evidence/degrees.
        var emittedEdges: Set<String> = []

        var hop = 1
        while hop <= k && !currentLayer.isEmpty {
            var nextLayer: [EntityID] = []
            for entity in currentLayer {
                for (neighbor, relationship) in graph.neighbors(of: entity) {
                    guard passesFilter(relationship) else { continue }
                    let edgeKey =
                        "\(relationship.source.raw)|\(relationship.target.raw)|\(relationship.relationType)"
                    if emittedEdges.insert(edgeKey).inserted {
                        result.relationships.append(relationship)
                    }
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor)
                        result.distances[neighbor] = hop
                        result.entities.append(neighbor)
                        nextLayer.append(neighbor)
                    }
                }
            }
            currentLayer = nextLayer
            hop += 1
        }
        return result
    }

    /// Enumerate simple paths from `source` to `target` up to `maxDepth` hops,
    /// capped at `maxPaths`.
    public func findAllPaths(_ graph: KnowledgeGraph, from source: EntityID, to target: EntityID) -> [[EntityID]] {
        var paths: [[EntityID]] = []
        guard graph.contains(source), graph.contains(target) else { return paths }
        var visited: Set<EntityID> = []
        var current: [EntityID] = [source]
        pathDFS(graph, current: source, target: target, remaining: config.maxDepth,
                path: &current, visited: &visited, paths: &paths)
        return paths
    }

    private func pathDFS(
        _ graph: KnowledgeGraph,
        current: EntityID,
        target: EntityID,
        remaining: Int,
        path: inout [EntityID],
        visited: inout Set<EntityID>,
        paths: inout [[EntityID]]
    ) {
        if paths.count >= config.maxPaths { return }
        if current == target {
            paths.append(path)
            return
        }
        // `<= 0` (not `== 0`) so a negative configured maxDepth yields no expansion.
        if remaining <= 0 { return }
        visited.insert(current)
        // Paths are entity-only, so collapse parallel edges (multiple relation
        // types between the same nodes) to one neighbor to avoid duplicate paths.
        var uniqueNeighbors: [EntityID] = []
        var seenNeighbors: Set<EntityID> = []
        for (neighbor, relationship) in graph.neighbors(of: current) where passesFilter(relationship) {
            if seenNeighbors.insert(neighbor).inserted { uniqueNeighbors.append(neighbor) }
        }
        for neighbor in uniqueNeighbors where !visited.contains(neighbor) {
            path.append(neighbor)
            pathDFS(graph, current: neighbor, target: target, remaining: remaining - 1,
                    path: &path, visited: &visited, paths: &paths)
            path.removeLast()
        }
        visited.remove(current)
    }
}
