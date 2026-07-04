import Testing

@testable import GraphRAG

// MARK: - Text chunking

@Test func chunkerProducesOverlappingChunks() throws {
    let text = String(
        repeating:
            "The quick brown fox jumps over the lazy dog. Knowledge graphs connect entities. ",
        count: 20)
    let chunker = HierarchicalChunker(minChunkSize: 10)
    let spans = chunker.chunkSpans(text, chunkSize: 200, overlap: 50)

    #expect(spans.count > 1)
    // Offsets are ordered and within bounds.
    for span in spans {
        #expect(span.startOffset >= 0)
        #expect(span.endOffset <= text.count)
        #expect(span.startOffset < span.endOffset)
    }
    // Consecutive chunks overlap (next start is before previous end).
    for i in 1..<spans.count {
        #expect(spans[i].startOffset < spans[i - 1].endOffset)
    }
}

@Test func textProcessorChunksShortDocument() throws {
    let processor = try TextProcessor(chunkSize: 1000, chunkOverlap: 100)
    let doc = Document(
        id: "doc1", title: "T",
        content: "Ada Lovelace collaborated with Charles Babbage on the Analytical Engine.")
    let chunks = processor.chunk(doc)
    #expect(chunks.count == 1)
    #expect(chunks[0].id == ChunkID("doc1_0"))
    #expect(chunks[0].metadata.wordCount > 0)
}

// MARK: - Keyword extraction

@Test func tfidfExtractsContentKeywords() {
    let extractor = TfIdfKeywordExtractor()
    let keywords = extractor.extractKeywordStrings(
        "Knowledge graphs represent entities and relationships between entities.", topK: 3)
    #expect(!keywords.isEmpty)
    // Stopwords like "and"/"between" must be filtered out.
    #expect(!keywords.contains("and"))
}

// MARK: - BM25

@Test func bm25RanksRelevantDocumentFirst() {
    var bm25 = BM25Retriever()
    bm25.index(id: "a", content: "Graph databases store nodes and edges efficiently.")
    bm25.index(id: "b", content: "Cooking recipes for delicious pasta dishes.")
    bm25.index(id: "c", content: "Knowledge graphs use nodes edges and graph traversal.")

    let results = bm25.search("graph nodes edges", limit: 3)
    #expect(!results.isEmpty)
    // A graph-related doc should outrank the cooking doc.
    #expect(results.first?.id == "a" || results.first?.id == "c")
    #expect(!results.contains { $0.id == "b" && $0.score > (results.first?.score ?? 0) })
}

// MARK: - Vector store & embeddings

@Test func cosineSimilarityBasics() {
    #expect(abs(cosineSimilarity([1, 0], [1, 0]) - 1.0) < 1e-6)
    #expect(abs(cosineSimilarity([1, 0], [0, 1])) < 1e-6)
}

@Test func hashEmbedderIsDeterministicAndDimensioned() {
    let embedder = HashEmbedder(dimension: 64)
    let a = embedder.embedSync("knowledge graph retrieval")
    let b = embedder.embedSync("knowledge graph retrieval")
    #expect(a == b)
    #expect(a.count == 64)
}

@Test func vectorStoreReturnsNearestNeighbor() {
    let embedder = HashEmbedder(dimension: 128)
    var store = InMemoryVectorStore()
    store.add(id: "graphs", vector: embedder.embedSync("graphs nodes edges entities"))
    store.add(id: "cooking", vector: embedder.embedSync("cooking pasta tomato recipe"))

    let query = embedder.embedSync("entities and nodes in graphs")
    let results = store.search(query, k: 2)
    #expect(results.first?.id == "graphs")
}

// MARK: - Knowledge graph

@Test func knowledgeGraphStoresEntitiesAndNeighbors() {
    var graph = KnowledgeGraph()
    let ada = Entity(id: "person_ada", name: "Ada Lovelace", entityType: "PERSON")
    let babbage = Entity(id: "person_babbage", name: "Charles Babbage", entityType: "PERSON")
    graph.addEntity(ada)
    graph.addEntity(babbage)
    graph.addRelationship(
        Relationship(source: ada.id, target: babbage.id, relationType: "COLLEAGUE_OF"))

    #expect(graph.entityCount == 2)
    #expect(graph.relationshipCount == 1)
    let neighbors = graph.neighbors(of: ada.id)
    #expect(neighbors.contains { $0.neighbor == babbage.id })
    // Bidirectional lookup.
    #expect(graph.neighbors(of: babbage.id).contains { $0.neighbor == ada.id })
}

@Test func knowledgeGraphMergesDuplicateRelationships() {
    var graph = KnowledgeGraph()
    graph.addEntity(Entity(id: "a", name: "A", entityType: "X"))
    graph.addEntity(Entity(id: "b", name: "B", entityType: "X"))
    graph.addRelationship(Relationship(source: "a", target: "b", relationType: "R", confidence: 0.5))
    graph.addRelationship(Relationship(source: "a", target: "b", relationType: "R", confidence: 0.9))
    #expect(graph.relationshipCount == 1)
    #expect(graph.relationships[0].confidence == 0.9)
}

// MARK: - Graph algorithms

@Test func pageRankScoresSumToOneAndRankHub() {
    var graph = KnowledgeGraph()
    for name in ["a", "b", "c", "hub"] {
        graph.addEntity(Entity(id: EntityID(name), name: name, entityType: "X"))
    }
    // Everyone points to the hub.
    graph.addRelationship(Relationship(source: "a", target: "hub", relationType: "R"))
    graph.addRelationship(Relationship(source: "b", target: "hub", relationType: "R"))
    graph.addRelationship(Relationship(source: "c", target: "hub", relationType: "R"))

    let scores = PageRank().compute(graph)
    let total = scores.values.reduce(0, +)
    #expect(abs(total - 1.0) < 1e-6)
    let hub = scores[EntityID("hub")] ?? 0
    #expect(hub > (scores[EntityID("a")] ?? 0))
}

@Test func bfsTraversalRespectsDepth() {
    var graph = KnowledgeGraph()
    for name in ["a", "b", "c", "d"] {
        graph.addEntity(Entity(id: EntityID(name), name: name, entityType: "X"))
    }
    graph.addRelationship(Relationship(source: "a", target: "b", relationType: "R", confidence: 1))
    graph.addRelationship(Relationship(source: "b", target: "c", relationType: "R", confidence: 1))
    graph.addRelationship(Relationship(source: "c", target: "d", relationType: "R", confidence: 1))

    let traversal = GraphTraversal(config: TraversalConfig(maxDepth: 2, minRelationshipStrength: 0.5))
    let result = traversal.bfs(graph, from: "a")
    #expect(result.distances[EntityID("a")] == 0)
    #expect(result.distances[EntityID("b")] == 1)
    #expect(result.distances[EntityID("c")] == 2)
    // 'd' is at depth 3, beyond maxDepth.
    #expect(result.distances[EntityID("d")] == nil)
}

@Test func analyticsDegreeAndComponents() {
    var graph = KnowledgeGraph()
    for name in ["a", "b", "c"] {
        graph.addEntity(Entity(id: EntityID(name), name: name, entityType: "X"))
    }
    graph.addRelationship(Relationship(source: "a", target: "b", relationType: "R"))
    let analytics = GraphAnalytics(graph)
    // 'a' connects to 'b' out of 2 possible -> 0.5.
    #expect(abs(analytics.degreeCentrality("a") - 0.5) < 1e-6)
    // 'a'+'b' connected, 'c' isolated -> 2 components.
    #expect(analytics.connectedComponents().count == 2)
}

// MARK: - Pattern extraction

@Test func patternExtractorFindsPeople() async throws {
    let extractor = PatternEntityExtractor(minConfidence: 0.5)
    let chunk = TextChunk(
        id: "c0", documentID: "d0",
        content: "Ada Lovelace worked with Charles Babbage in London.",
        startOffset: 0, endOffset: 0)
    let (entities, _) = try await extractor.extract(from: chunk)
    let names = Set(entities.map(\.name))
    #expect(names.contains("Ada Lovelace"))
    #expect(names.contains("Charles Babbage"))
}

// MARK: - End-to-end pipeline

@Test func endToEndBuildAndAskWithoutLLM() async throws {
    let rag = try GraphRAGBuilder()
        .withChunkSize(400)
        .withChunkOverlap(50)
        .withTopK(3)
        .build()

    await rag.addDocument(
        text: """
            Ada Lovelace was an English mathematician. She collaborated with Charles Babbage
            on the Analytical Engine, an early mechanical general-purpose computer. Ada is
            often regarded as the first computer programmer.
            """)
    await rag.addDocument(
        text: "Pasta is cooked in boiling water with salt. Tomato sauce is a common topping.")

    try await rag.build()

    let stats = await rag.stats()
    #expect(stats.documentCount == 2)
    #expect(stats.chunkCount >= 2)
    #expect(stats.entityCount > 0)

    let answer = try await rag.ask("Who worked on the Analytical Engine?")
    #expect(!answer.text.isEmpty)
    #expect(!answer.sources.isEmpty)
    // The relevant (computing) chunk should be retrieved over the pasta chunk.
    #expect(answer.text.lowercased().contains("babbage") || answer.text.lowercased().contains("ada"))
}

@Test func askBeforeBuildThrows() async throws {
    let rag = try GraphRAGBuilder().build()
    await rag.addDocument(text: "Some content about graphs and entities.")
    await #expect(throws: GraphRAGError.self) {
        _ = try await rag.ask("anything")
    }
}

// MARK: - Review regressions

@Test func negativeChunkOverlapRejected() {
    #expect(throws: GraphRAGError.self) {
        _ = try TextProcessor(chunkSize: 100, chunkOverlap: -10)
    }
}

@Test func replacingDocumentRemovesStaleChunks() async throws {
    let rag = try GraphRAGBuilder().withChunkSize(500).withChunkOverlap(50).build()
    let id = DocumentID("fixed-id")
    await rag.addDocument(Document(id: id, title: "v1", content: "First version about apples."))
    await rag.addDocument(Document(id: id, title: "v2", content: "Second version about oranges."))
    try await rag.build()
    let stats = await rag.stats()
    // Only the replacement's chunk(s) should remain, not both versions'.
    #expect(stats.documentCount == 1)
    #expect(stats.chunkCount == 1)
    let answer = try await rag.ask("oranges")
    #expect(!answer.text.lowercased().contains("apples"))
}

@Test func dfsDoesNotRecordEdgesBeyondMaxDepth() {
    var graph = KnowledgeGraph()
    for name in ["a", "b", "c", "d"] {
        graph.addEntity(Entity(id: EntityID(name), name: name, entityType: "X"))
    }
    graph.addRelationship(Relationship(source: "a", target: "b", relationType: "R", confidence: 1))
    graph.addRelationship(Relationship(source: "b", target: "c", relationType: "R", confidence: 1))
    graph.addRelationship(Relationship(source: "c", target: "d", relationType: "R", confidence: 1))

    let traversal = GraphTraversal(config: TraversalConfig(maxDepth: 2, minRelationshipStrength: 0.5))
    let result = traversal.dfs(graph, from: "a")
    let visited = Set(result.entities)
    #expect(!visited.contains(EntityID("d")))
    // Every recorded edge must connect two visited nodes.
    for rel in result.relationships {
        #expect(visited.contains(rel.source))
        #expect(visited.contains(rel.target))
    }
}

@Test func negativeTopKDoesNotCrashSearch() async throws {
    let config = Config(topKResults: -5)
    let rag = try GraphRAGBuilder().withConfig(config).build()
    await rag.addDocument(text: "Graphs connect entities and relationships.")
    try await rag.build()
    let answer = try await rag.ask("graphs")
    // Should degrade to the no-results answer rather than trapping.
    #expect(answer.sources.isEmpty)
}
