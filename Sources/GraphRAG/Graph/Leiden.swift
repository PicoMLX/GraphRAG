// Leiden.swift
// Community detection over the knowledge graph, ported from graphrag-rs
// `graph::leiden`.
//
// The Rust implementation is a simplified single-level Leiden: greedy modularity
// local-moving followed by one refinement pass that splits internally
// disconnected communities. This port keeps that structure but makes it
// deterministic (stable node ordering) and *weighted* — it uses each
// relationship's confidence as an edge weight, since the `KnowledgeGraph`
// carries them (the Rust version ignored weights). Only configuration that
// actually affects the result is exposed.

import Foundation

/// Tunables for `LeidenCommunityDetector`.
public struct LeidenConfig: Sendable {
    /// Higher resolution → more, smaller communities (default 1.0).
    public var resolution: Double
    /// Maximum local-moving passes (default 100).
    public var maxIterations: Int
    /// Stop when a pass improves modularity by less than this (default 1e-6).
    public var minModularityGain: Double

    public init(resolution: Double = 1.0, maxIterations: Int = 100, minModularityGain: Double = 1e-6) {
        self.resolution = resolution
        self.maxIterations = maxIterations
        self.minModularityGain = minModularityGain
    }
}

/// A detected community: a contiguous integer id and its member entities.
public struct Community: Sendable, Equatable, Identifiable {
    public var id: Int
    public var members: [EntityID]

    public init(id: Int, members: [EntityID]) {
        self.id = id
        self.members = members
    }

    public var size: Int { members.count }
}

/// The output of community detection.
public struct CommunityDetectionResult: Sendable {
    /// Communities ordered by id, each with its members (in node order).
    public var communities: [Community]
    /// Map from entity to its community id.
    public var assignment: [EntityID: Int]
    /// Modularity of the final partition.
    public var modularity: Double

    public init(communities: [Community], assignment: [EntityID: Int], modularity: Double) {
        self.communities = communities
        self.assignment = assignment
        self.modularity = modularity
    }

    public var communityCount: Int { communities.count }
}

/// Weighted, deterministic Leiden community detection.
public struct LeidenCommunityDetector: Sendable {
    public var config: LeidenConfig

    public init(config: LeidenConfig = LeidenConfig()) {
        self.config = config
    }

    public func detect(_ graph: KnowledgeGraph) -> CommunityDetectionResult {
        let nodes = graph.entities.map(\.id)
        let n = nodes.count
        guard n > 0 else {
            return CommunityDetectionResult(communities: [], assignment: [:], modularity: 0)
        }

        var indexOf: [EntityID: Int] = [:]
        for (i, id) in nodes.enumerated() { indexOf[id] = i }

        // Build an undirected, weighted adjacency (parallel edges summed, self
        // loops dropped, non-positive weights skipped).
        var adjacency: [[(node: Int, weight: Double)]] = Array(repeating: [], count: n)
        var mergedWeight: [Int: [Int: Double]] = [:]
        for rel in graph.relationships {
            guard let s = indexOf[rel.source], let t = indexOf[rel.target], s != t else { continue }
            let w = Double(rel.confidence)
            guard w > 0 else { continue }
            mergedWeight[s, default: [:]][t, default: 0] += w
            mergedWeight[t, default: [:]][s, default: 0] += w
        }
        // Build adjacency by node index and sorted neighbor id so both the
        // neighbor ordering and the floating-point degree summation are
        // deterministic across runs (dictionary iteration order is randomized).
        var degree = [Double](repeating: 0, count: n)
        for i in 0..<n {
            guard let neighbors = mergedWeight[i] else { continue }
            for (j, w) in neighbors.sorted(by: { $0.key < $1.key }) {
                adjacency[i].append((j, w))
                degree[i] += w
            }
        }
        let twoM = degree.reduce(0, +)

        // No edges → every node is its own community.
        guard twoM > 0 else {
            return singletons(nodes)
        }

        // Phase 1: greedy modularity local moving.
        var communityOf = Array(0..<n)
        var sigmaTot = degree  // Σ_tot per community id (ids stay in 0..<n here)
        var previousModularity = modularity(communityOf, adjacency, degree, twoM)

        // `maxIterations == 0` means "no local moving" (refinement-only /
        // singleton baseline); clamp negatives to 0 so the range never traps.
        for _ in 0..<max(0, config.maxIterations) {
            var moved = false
            for i in 0..<n {
                let ci = communityOf[i]
                let ki = degree[i]
                // Remove i from its community.
                sigmaTot[ci] -= ki

                // Weight from i to each candidate (neighbor) community.
                var weightToCommunity: [Int: Double] = [:]
                for edge in adjacency[i] {
                    weightToCommunity[communityOf[edge.node], default: 0] += edge.weight
                }

                // Score staying in ci, then every neighbor community; keep the
                // best (ties: prefer current, then lowest id — deterministic).
                //
                // We intentionally do NOT score the empty-singleton candidate
                // (Σ_tot = 0) here, so a connected over-merge from an earlier pass
                // can't be split back into a singleton by local moving. This keeps
                // the port aligned with the upstream simplified single-level Leiden
                // (disconnected communities are still split in the refinement pass)
                // and avoids introducing an unverifiable change to this
                // deterministic core — full Leiden singleton refinement is a
                // deliberate non-goal for this port.
                var bestCommunity = ci
                var bestScore = (weightToCommunity[ci] ?? 0) - config.resolution * ki * sigmaTot[ci] / twoM
                for (comm, wIn) in weightToCommunity.sorted(by: { $0.key < $1.key }) where comm != ci {
                    let score = wIn - config.resolution * ki * sigmaTot[comm] / twoM
                    if score > bestScore + 1e-12 {
                        bestScore = score
                        bestCommunity = comm
                    }
                }

                communityOf[i] = bestCommunity
                sigmaTot[bestCommunity] += ki
                if bestCommunity != ci { moved = true }
            }

            let currentModularity = modularity(communityOf, adjacency, degree, twoM)
            if !moved || currentModularity - previousModularity < config.minModularityGain {
                previousModularity = currentModularity
                break
            }
            previousModularity = currentModularity
        }

        // Phase 2: refinement — split internally disconnected communities.
        refine(&communityOf, adjacency: adjacency, n: n)

        // Renumber to contiguous ids ordered by first member appearance.
        return finalize(communityOf, nodes: nodes, adjacency: adjacency, degree: degree, twoM: twoM)
    }

    // MARK: - Internals

    private func singletons(_ nodes: [EntityID]) -> CommunityDetectionResult {
        var communities: [Community] = []
        var assignment: [EntityID: Int] = [:]
        for (i, id) in nodes.enumerated() {
            communities.append(Community(id: i, members: [id]))
            assignment[id] = i
        }
        return CommunityDetectionResult(communities: communities, assignment: assignment, modularity: 0)
    }

    private func refine(
        _ communityOf: inout [Int], adjacency: [[(node: Int, weight: Double)]], n: Int
    ) {
        var members: [Int: [Int]] = [:]
        for i in 0..<n { members[communityOf[i], default: []].append(i) }
        var nextId = (communityOf.max() ?? -1) + 1

        for community in members.keys.sorted() {
            let nodes = members[community]!
            let nodeSet = Set(nodes)
            var visited: Set<Int> = []
            var components: [[Int]] = []
            for start in nodes where !visited.contains(start) {
                var component: [Int] = []
                var queue = [start]
                visited.insert(start)
                var head = 0
                while head < queue.count {
                    let u = queue[head]
                    head += 1
                    component.append(u)
                    for edge in adjacency[u]
                    where nodeSet.contains(edge.node) && !visited.contains(edge.node) {
                        visited.insert(edge.node)
                        queue.append(edge.node)
                    }
                }
                components.append(component)
            }
            // Keep the first component under the original id; give the rest new ids.
            if components.count > 1 {
                for componentIndex in 1..<components.count {
                    for u in components[componentIndex] { communityOf[u] = nextId }
                    nextId += 1
                }
            }
        }
    }

    private func finalize(
        _ communityOf: [Int], nodes: [EntityID],
        adjacency: [[(node: Int, weight: Double)]], degree: [Double], twoM: Double
    ) -> CommunityDetectionResult {
        var canonical: [Int: Int] = [:]
        var finalOf = [Int](repeating: 0, count: nodes.count)
        var nextId = 0
        for i in 0..<nodes.count {
            let c = communityOf[i]
            if let mapped = canonical[c] {
                finalOf[i] = mapped
            } else {
                canonical[c] = nextId
                finalOf[i] = nextId
                nextId += 1
            }
        }

        var membersById: [[EntityID]] = Array(repeating: [], count: nextId)
        var assignment: [EntityID: Int] = [:]
        for i in 0..<nodes.count {
            membersById[finalOf[i]].append(nodes[i])
            assignment[nodes[i]] = finalOf[i]
        }
        let communities = membersById.enumerated().map { Community(id: $0.offset, members: $0.element) }
        let q = modularity(finalOf, adjacency, degree, twoM)
        return CommunityDetectionResult(
            communities: communities, assignment: assignment, modularity: q)
    }

    /// Weighted Newman modularity of a partition.
    private func modularity(
        _ communityOf: [Int], _ adjacency: [[(node: Int, weight: Double)]],
        _ degree: [Double], _ twoM: Double
    ) -> Double {
        guard twoM > 0 else { return 0 }
        var internalWeight: [Int: Double] = [:]  // Σ_in per community (edges counted twice)
        var totalDegree: [Int: Double] = [:]
        for i in 0..<communityOf.count {
            let c = communityOf[i]
            totalDegree[c, default: 0] += degree[i]
            for edge in adjacency[i] where communityOf[edge.node] == c {
                internalWeight[c, default: 0] += edge.weight
            }
        }
        // Sum in sorted community-id order so the (non-associative) floating-point
        // total is identical across runs.
        var q = 0.0
        for c in totalDegree.keys.sorted() {
            let tot = totalDegree[c]!
            let sin = internalWeight[c] ?? 0
            q += sin / twoM - config.resolution * (tot / twoM) * (tot / twoM)
        }
        return q
    }
}
