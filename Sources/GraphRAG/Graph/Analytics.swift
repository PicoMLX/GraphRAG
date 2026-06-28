// Analytics.swift
// Ported from graphrag-rs `graph::analytics`.
//
// Centrality measures treat the graph as undirected (edges connect both
// endpoints), matching the bidirectional neighbour semantics used elsewhere.

import Foundation

/// The three centrality scores for an entity.
public struct CentralityScores: Sendable, Equatable {
    public var degree: Float
    public var betweenness: Float
    public var closeness: Float

    public init(degree: Float = 0, betweenness: Float = 0, closeness: Float = 0) {
        self.degree = degree
        self.betweenness = betweenness
        self.closeness = closeness
    }
}

/// Graph-level and node-level structural metrics.
public struct GraphAnalytics: Sendable {
    private let graph: KnowledgeGraph
    private let nodes: [EntityID]
    private let adjacency: [EntityID: [EntityID]]

    public init(_ graph: KnowledgeGraph) {
        self.graph = graph
        let nodeList = graph.entities.map(\.id)
        self.nodes = nodeList
        let nodeSet = Set(nodeList)
        var adj: [EntityID: Set<EntityID>] = [:]
        for id in nodeList { adj[id] = [] }
        for rel in graph.relationships {
            // Only connect endpoints that are actual graph nodes.
            guard nodeSet.contains(rel.source), nodeSet.contains(rel.target) else { continue }
            adj[rel.source, default: []].insert(rel.target)
            adj[rel.target, default: []].insert(rel.source)
        }
        self.adjacency = adj.mapValues { Array($0) }
    }

    private func neighbors(_ id: EntityID) -> [EntityID] { adjacency[id] ?? [] }

    // MARK: - Degree

    /// Degree centrality: `degree / (n - 1)`, in `[0, 1]`.
    public func degreeCentrality(_ id: EntityID) -> Float {
        let n = nodes.count
        guard n > 1 else { return 0 }
        return Float(neighbors(id).count) / Float(n - 1)
    }

    // MARK: - Closeness

    /// Closeness centrality: reachable node count divided by total distance.
    public func closenessCentrality(_ id: EntityID) -> Float {
        let distances = bfsDistances(from: id)
        var total = 0
        var reachable = 0
        for (node, dist) in distances where node != id {
            total += dist
            reachable += 1
        }
        guard total > 0 else { return 0 }
        return Float(reachable) / Float(total)
    }

    // MARK: - Betweenness (Brandes, unweighted)

    /// Normalized betweenness centrality for every node, via Brandes' algorithm.
    public func betweennessCentrality() -> [EntityID: Float] {
        var betweenness: [EntityID: Double] = [:]
        for id in nodes { betweenness[id] = 0 }
        let n = nodes.count
        guard n > 2 else { return betweenness.mapValues { Float($0) } }

        for source in nodes {
            var stack: [EntityID] = []
            var predecessors: [EntityID: [EntityID]] = [:]
            var sigma: [EntityID: Double] = [:]
            var dist: [EntityID: Int] = [:]
            for id in nodes { sigma[id] = 0; dist[id] = -1; predecessors[id] = [] }
            sigma[source] = 1
            dist[source] = 0

            var queue: [EntityID] = [source]
            var head = 0
            while head < queue.count {
                let v = queue[head]; head += 1
                stack.append(v)
                for w in neighbors(v) {
                    if dist[w]! < 0 {
                        dist[w] = dist[v]! + 1
                        queue.append(w)
                    }
                    if dist[w]! == dist[v]! + 1 {
                        sigma[w]! += sigma[v]!
                        predecessors[w]!.append(v)
                    }
                }
            }

            var delta: [EntityID: Double] = [:]
            for id in nodes { delta[id] = 0 }
            while let w = stack.popLast() {
                for v in predecessors[w]! {
                    delta[v]! += (sigma[v]! / sigma[w]!) * (1 + delta[w]!)
                }
                if w != source { betweenness[w]! += delta[w]! }
            }
        }

        // Undirected: each pair counted twice; normalize to [0, 1].
        let norm = Double((n - 1) * (n - 2))
        var result: [EntityID: Float] = [:]
        for (id, value) in betweenness {
            result[id] = norm > 0 ? Float(value / norm) : 0
        }
        return result
    }

    /// Combined centrality scores for a single node.
    public func centrality(_ id: EntityID) -> CentralityScores {
        CentralityScores(
            degree: degreeCentrality(id),
            betweenness: betweennessCentrality()[id] ?? 0,
            closeness: closenessCentrality(id)
        )
    }

    // MARK: - Components

    /// The connected component (undirected) containing `start`.
    public func connectedComponent(containing start: EntityID) -> [EntityID] {
        guard graph.contains(start) else { return [] }
        var visited: Set<EntityID> = [start]
        var queue: [EntityID] = [start]
        var component: [EntityID] = []
        var head = 0
        while head < queue.count {
            let current = queue[head]; head += 1
            component.append(current)
            for neighbor in neighbors(current) where !visited.contains(neighbor) {
                visited.insert(neighbor)
                queue.append(neighbor)
            }
        }
        return component
    }

    /// All connected components of the graph.
    public func connectedComponents() -> [[EntityID]] {
        var visited: Set<EntityID> = []
        var components: [[EntityID]] = []
        for node in nodes where !visited.contains(node) {
            let component = connectedComponent(containing: node)
            for c in component { visited.insert(c) }
            components.append(component)
        }
        return components
    }

    // MARK: - Global

    /// Graph density: `2E / (n(n-1))`.
    public func density() -> Float {
        let n = nodes.count
        guard n > 1 else { return 0 }
        return Float(2 * graph.relationshipCount) / Float(n * (n - 1))
    }

    /// Local clustering coefficient: fraction of a node's neighbour pairs that
    /// are themselves connected.
    public func clusteringCoefficient(_ id: EntityID) -> Float {
        let ns = neighbors(id)
        let k = ns.count
        guard k > 1 else { return 0 }
        var links = 0
        for i in 0..<k {
            let iNeighbors = Set(neighbors(ns[i]))
            for j in (i + 1)..<k where iNeighbors.contains(ns[j]) {
                links += 1
            }
        }
        let possible = k * (k - 1) / 2
        return possible > 0 ? Float(links) / Float(possible) : 0
    }

    private func bfsDistances(from source: EntityID) -> [EntityID: Int] {
        var dist: [EntityID: Int] = [source: 0]
        var queue: [EntityID] = [source]
        var head = 0
        while head < queue.count {
            let current = queue[head]; head += 1
            let d = dist[current]!
            for neighbor in neighbors(current) where dist[neighbor] == nil {
                dist[neighbor] = d + 1
                queue.append(neighbor)
            }
        }
        return dist
    }
}
