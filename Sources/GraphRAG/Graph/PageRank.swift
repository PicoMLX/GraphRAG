// PageRank.swift
// Ported from graphrag-rs `graph::pagerank`.

import Foundation

/// Weighted PageRank over the knowledge graph's directed relationships.
public struct PageRank: Sendable {
    /// Probability of following a link vs. teleporting (default 0.85).
    public var dampingFactor: Double
    /// Maximum power iterations (default 100).
    public var maxIterations: Int
    /// L-infinity convergence threshold (default 1e-6).
    public var tolerance: Double

    public init(dampingFactor: Double = 0.85, maxIterations: Int = 100, tolerance: Double = 1e-6) {
        self.dampingFactor = dampingFactor
        self.maxIterations = maxIterations
        self.tolerance = tolerance
    }

    /// Compute a PageRank score in `[0, 1]` for each entity. Scores sum to 1.
    public func compute(_ graph: KnowledgeGraph) -> [EntityID: Double] {
        let nodes = graph.entities.map(\.id)
        let n = nodes.count
        guard n > 0 else { return [:] }
        if n == 1 { return [nodes[0]: 1.0] }

        var indexOf: [EntityID: Int] = [:]
        for (i, id) in nodes.enumerated() { indexOf[id] = i }

        // Incoming contributions: for each target i, list of (source j, weight).
        var incomingEdges: [[(source: Int, weight: Double)]] = Array(repeating: [], count: n)
        var outWeight = [Double](repeating: 0, count: n)
        for rel in graph.relationships {
            guard let s = indexOf[rel.source], let t = indexOf[rel.target] else { continue }
            let w = Double(max(rel.confidence, 0.0001))
            incomingEdges[t].append((s, w))
            outWeight[s] += w
        }

        // Clamp to a valid probability so a misconfigured factor can't produce a
        // negative teleport term (and negative scores).
        let d = min(max(dampingFactor, 0), 1)
        let teleport = (1.0 - d) / Double(n)
        var scores = [Double](repeating: 1.0 / Double(n), count: n)

        for _ in 0..<max(0, maxIterations) {
            // Dangling-node mass: nodes with no out-edges redistribute uniformly.
            var danglingMass = 0.0
            for i in 0..<n where outWeight[i] == 0 { danglingMass += scores[i] }
            let danglingShare = d * danglingMass / Double(n)

            var next = [Double](repeating: teleport + danglingShare, count: n)
            for i in 0..<n {
                var sum = 0.0
                for edge in incomingEdges[i] {
                    sum += (edge.weight / outWeight[edge.source]) * scores[edge.source]
                }
                next[i] += d * sum
            }

            var delta = 0.0
            for i in 0..<n { delta = max(delta, abs(next[i] - scores[i])) }
            scores = next
            if delta < tolerance { break }
        }

        // Normalize to a probability distribution.
        let total = scores.reduce(0, +)
        if total > 0 {
            for i in 0..<n { scores[i] /= total }
        }

        var result: [EntityID: Double] = [:]
        result.reserveCapacity(n)
        for (i, id) in nodes.enumerated() { result[id] = scores[i] }
        return result
    }

    /// Top-`k` entities by PageRank score, highest first.
    public func topEntities(_ graph: KnowledgeGraph, k: Int) -> [(id: EntityID, score: Double)] {
        guard k > 0 else { return [] }
        let scores = compute(graph)
        return scores.sorted { lhs, rhs in
            if lhs.value == rhs.value { return lhs.key.raw < rhs.key.raw }
            return lhs.value > rhs.value
        }
        .prefix(k)
        .map { (id: $0.key, score: $0.value) }
    }
}
