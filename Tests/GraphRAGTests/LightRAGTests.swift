import Testing

@testable import GraphRAG

// MARK: - Test doubles

private struct MockLLM: LanguageModel {
    let response: String
    func complete(_ prompt: String, params: GenerationParams) async throws -> String { response }
    func isAvailable() async -> Bool { true }
    var modelInfo: ModelInfo { ModelInfo(name: "mock") }
}

private struct MockSearcher: SemanticSearcher {
    let results: [LightRAGResult]
    func search(_ query: String, topK: Int) async throws -> [LightRAGResult] {
        Array(results.prefix(topK))
    }
}

private func triangleGraph() -> KnowledgeGraph {
    var graph = KnowledgeGraph()
    let names = ["a1", "a2", "a3", "b1", "b2", "b3"]
    for name in names {
        graph.addEntity(Entity(id: EntityID(name), name: name, entityType: "X"))
    }
    func link(_ x: String, _ y: String) {
        graph.addRelationship(
            Relationship(source: EntityID(x), target: EntityID(y), relationType: "R", confidence: 1))
    }
    // Two disconnected triangles.
    link("a1", "a2"); link("a2", "a3"); link("a1", "a3")
    link("b1", "b2"); link("b2", "b3"); link("b1", "b3")
    return graph
}

// MARK: - Leiden

@Test func leidenSeparatesDisconnectedClusters() {
    let result = LeidenCommunityDetector().detect(triangleGraph())
    #expect(result.communityCount == 2)
    // Each triangle's nodes share a community.
    let a = result.assignment[EntityID("a1")]
    #expect(result.assignment[EntityID("a2")] == a)
    #expect(result.assignment[EntityID("a3")] == a)
    let b = result.assignment[EntityID("b1")]
    #expect(result.assignment[EntityID("b2")] == b)
    #expect(result.assignment[EntityID("b3")] == b)
    #expect(a != b)
    #expect(result.modularity > 0)
}

@Test func leidenEmptyGraphIsEmpty() {
    let result = LeidenCommunityDetector().detect(KnowledgeGraph())
    #expect(result.communityCount == 0)
    #expect(result.modularity == 0)
}

@Test func leidenNoEdgesYieldsSingletons() {
    var graph = KnowledgeGraph()
    for name in ["x", "y", "z"] {
        graph.addEntity(Entity(id: EntityID(name), name: name, entityType: "T"))
    }
    let result = LeidenCommunityDetector().detect(graph)
    #expect(result.communityCount == 3)
}

@Test func leidenIsDeterministic() {
    let graph = triangleGraph()
    let a = LeidenCommunityDetector().detect(graph)
    let b = LeidenCommunityDetector().detect(graph)
    #expect(a.assignment == b.assignment)
}

// MARK: - Keyword extraction

@Test func keywordFallbackTokenizesQuery() async {
    let extractor = KeywordExtractor(model: nil)
    let keywords = await extractor.extract("Find the quantum computing project")
    #expect(keywords.highLevel.isEmpty)
    #expect(keywords.lowLevel.contains("quantum"))
    #expect(!keywords.lowLevel.contains("the"))  // shorter than 4 chars
}

@Test func keywordLLMParsesDualLevels() async {
    let json = #"{"high_level": ["collaboration"], "low_level": ["Alice", "Bob"]}"#
    let extractor = KeywordExtractor(model: MockLLM(response: json))
    let keywords = await extractor.extract("How did Alice and Bob collaborate?")
    #expect(keywords.highLevel == ["collaboration"])
    #expect(keywords.lowLevel == ["Alice", "Bob"])
}

// MARK: - Dual-level retrieval / merge

@Test func dualRetrievalInterleavesAndDedupsWithinLevel() async throws {
    // A store that returns a duplicate id within its own results is deduped.
    let high = MockSearcher(results: [
        LightRAGResult(id: "h1", content: "theme one", score: 0.9),
        LightRAGResult(id: "h1", content: "theme one again", score: 0.4),
    ])
    let low = MockSearcher(results: [
        LightRAGResult(id: "l1", content: "detail one", score: 0.8)
    ])
    let extractor = KeywordExtractor(
        model: MockLLM(response: #"{"high_level":["t"],"low_level":["d"]}"#))
    let retriever = DualLevelRetriever(
        keywordExtractor: extractor, highLevelStore: high, lowLevelStore: low)

    let results = try await retriever.retrieve("query", topK: 5)
    let ids = results.mergedChunks.map(\.id)
    #expect(ids.first == "h1")
    #expect(ids.filter { $0 == "h1" }.count == 1)  // within-level dedup
    #expect(Set(ids) == ["h1", "l1"])
}

@Test func dualRetrievalKeepsCollidingCrossLevelIds() async throws {
    // A real chunk id equal to a community id must survive on both levels — the
    // high (community) and low (chunk) hits are distinct results and neither
    // should silently evict the other.
    let high = MockSearcher(results: [
        LightRAGResult(id: "community_0", content: "summary", score: 0.9)
    ])
    let low = MockSearcher(results: [
        LightRAGResult(id: "community_0", content: "chunk text", score: 0.8)
    ])
    let extractor = KeywordExtractor(
        model: MockLLM(response: #"{"high_level":["t"],"low_level":["d"]}"#))
    let retriever = DualLevelRetriever(
        keywordExtractor: extractor, highLevelStore: high, lowLevelStore: low)

    let results = try await retriever.retrieve("query", topK: 5)
    #expect(results.mergedChunks.count == 2)
    #expect(results.mergedChunks.filter { $0.id == "community_0" }.count == 2)
}

@Test func communityGroundsToMemberChunks() throws {
    var graph = triangleGraph()
    let chunk = TextChunk(
        id: ChunkID("c1"), documentID: DocumentID("d"), content: "about a1",
        startOffset: 0, endOffset: 8, entities: [EntityID("a1")])
    graph.addChunk(chunk)
    let communities = LeidenCommunityDetector().detect(graph)
    let a1Community = try #require(
        communities.communities.first { $0.members.contains(EntityID("a1")) })
    let sources = LightRAG.communitySourceChunks(a1Community, graph: graph)
    #expect(sources.contains("c1"))
}

@Test func dualRetrievalWeightedOrdersByWeightedScore() async throws {
    let high = MockSearcher(results: [LightRAGResult(id: "h", content: "h", score: 1.0)])
    let low = MockSearcher(results: [LightRAGResult(id: "l", content: "l", score: 1.0)])
    let extractor = KeywordExtractor(
        model: MockLLM(response: #"{"high_level":["t"],"low_level":["d"]}"#))
    // highLevelWeight 0.6 > lowLevelWeight 0.4, equal base scores → high first.
    let retriever = DualLevelRetriever(
        keywordExtractor: extractor, highLevelStore: high, lowLevelStore: low,
        config: DualRetrievalConfig(mergeStrategy: .weighted))
    let results = try await retriever.retrieve("query", topK: 5)
    #expect(results.mergedChunks.map(\.id) == ["h", "l"])
}

// MARK: - End-to-end via GraphRAG

@Test func lightRAGEndToEndOffline() async throws {
    let rag = try GraphRAGBuilder().withChunkSize(400).withChunkOverlap(50).withTopK(3).build()
    await rag.addDocument(
        text: """
            Ada Lovelace collaborated with Charles Babbage on the Analytical Engine,
            an early mechanical general-purpose computer.
            """)
    try await rag.build()

    let engine = try await rag.lightRAG()
    let communities = engine.detectCommunities()
    #expect(communities.communityCount >= 1)

    let answer = try await engine.ask("Who worked on the Analytical Engine?")
    #expect(!answer.text.isEmpty)
    #expect(!answer.sources.isEmpty)
}
