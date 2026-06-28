// Chunking.swift
// Ported from graphrag-rs `text::chunking` (HierarchicalChunker) and the
// `TextProcessor` API in `text::mod`.
//
// The Rust implementation works on UTF-8 byte indices and guards every slice
// with `is_char_boundary`. Swift's `Character` (extended grapheme cluster) is
// always a valid boundary, so this port operates over a `[Character]` array and
// measures sizes/offsets in characters. For typical text this matches the byte
// behaviour while remaining Unicode-safe by construction.

import Foundation

/// A chunk's content together with its character offsets in the source text.
public struct ChunkSpan: Sendable, Equatable {
    public var content: String
    public var startOffset: Int
    public var endOffset: Int

    public init(content: String, startOffset: Int, endOffset: Int) {
        self.content = content
        self.startOffset = startOffset
        self.endOffset = endOffset
    }
}

/// Recursive, separator-aware chunker.
///
/// Splits on a hierarchy of separators (paragraph → line → sentence → clause →
/// word), preferring the "highest" separator that yields a boundary past the
/// first quarter of the window.
public struct HierarchicalChunker: Sendable {
    /// Ordered, most-significant-first list of separators.
    public var separators: [String]
    /// Chunks whose trimmed length is below this are discarded.
    public var minChunkSize: Int

    public static let defaultSeparators: [String] = [
        "\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", "",
    ]

    public init(separators: [String] = HierarchicalChunker.defaultSeparators, minChunkSize: Int = 50) {
        self.separators = separators
        self.minChunkSize = minChunkSize
    }

    public func withSeparators(_ separators: [String]) -> HierarchicalChunker {
        HierarchicalChunker(separators: separators, minChunkSize: minChunkSize)
    }

    public func withMinSize(_ size: Int) -> HierarchicalChunker {
        HierarchicalChunker(separators: separators, minChunkSize: size)
    }

    /// Split `text` into chunk strings of approximately `chunkSize` characters,
    /// overlapping consecutive chunks by `overlap` characters.
    public func chunkText(_ text: String, chunkSize: Int, overlap: Int) -> [String] {
        chunkSpans(text, chunkSize: chunkSize, overlap: overlap).map(\.content)
    }

    /// Like `chunkText` but also returns character offsets for each chunk.
    public func chunkSpans(_ text: String, chunkSize: Int, overlap: Int) -> [ChunkSpan] {
        let chars = Array(text)
        let n = chars.count
        guard n > 0, chunkSize > 0 else { return [] }

        var spans: [ChunkSpan] = []
        var start = 0

        while start < n {
            var end = min(start + chunkSize, n)

            // Final chunk: take the remainder.
            if end >= n {
                let slice = chars[start..<n]
                if trimmedCount(slice) >= minChunkSize || spans.isEmpty {
                    spans.append(makeSpan(slice, start: start, end: n))
                }
                break
            }

            let optimalEnd = findOptimalBoundary(chars, start: start, maxEnd: end)
            if optimalEnd > start { end = optimalEnd }

            let slice = chars[start..<end]
            if trimmedCount(slice) >= minChunkSize {
                spans.append(makeSpan(slice, start: start, end: end))
            }

            // Advance with overlap, snapped back to a word boundary.
            var nextStart = max(0, end - overlap)
            nextStart = findWordBoundaryBackward(chars, pos: nextStart)
            // Guarantee forward progress.
            if nextStart <= start { nextStart = end }
            start = nextStart
        }

        return spans
    }

    // MARK: - Boundary helpers

    private func makeSpan(_ slice: ArraySlice<Character>, start: Int, end: Int) -> ChunkSpan {
        ChunkSpan(content: String(slice), startOffset: start, endOffset: end)
    }

    private func trimmedCount(_ slice: ArraySlice<Character>) -> Int {
        String(slice).trimmingCharacters(in: .whitespacesAndNewlines).count
    }

    /// Find the best split point in `chars[start..<maxEnd]` by walking the
    /// separator hierarchy and taking the last occurrence that falls past the
    /// first quarter of the window.
    func findOptimalBoundary(_ chars: [Character], start: Int, maxEnd: Int) -> Int {
        let rangeLen = maxEnd - start
        guard rangeLen > 0 else { return maxEnd }
        let quarter = rangeLen / 4

        for separator in separators where !separator.isEmpty {
            let sep = Array(separator)
            if let matchStart = lastRange(of: sep, in: chars, start: start, end: maxEnd) {
                let boundary = matchStart + sep.count
                if boundary > start + quarter {
                    return boundary
                }
            }
        }
        return findWordBoundaryBackward(chars, pos: maxEnd)
    }

    /// Largest index `p <= pos` such that the character before `p` is whitespace.
    func findWordBoundaryBackward(_ chars: [Character], pos: Int) -> Int {
        var p = min(pos, chars.count)
        while p > 0 {
            if chars[p - 1].isWhitespace { return p }
            p -= 1
        }
        return 0
    }

    /// Last start-index of `needle` within `chars[start..<end]`, or nil.
    private func lastRange(of needle: [Character], in chars: [Character], start: Int, end: Int) -> Int? {
        guard !needle.isEmpty, end - start >= needle.count else { return nil }
        var i = end - needle.count
        while i >= start {
            var matched = true
            for j in 0..<needle.count where chars[i + j] != needle[j] {
                matched = false
                break
            }
            if matched { return i }
            i -= 1
        }
        return nil
    }
}

/// High-level text-processing facade mirroring the Rust `TextProcessor`.
public struct TextProcessor: Sendable {
    public var chunkSize: Int
    public var chunkOverlap: Int
    private let chunker: HierarchicalChunker
    private let keywordExtractor: TfIdfKeywordExtractor

    public init(chunkSize: Int = 1000, chunkOverlap: Int = 200) throws {
        guard chunkSize > 0 else {
            throw GraphRAGError.config(message: "chunk_size must be > 0")
        }
        guard chunkOverlap < chunkSize else {
            throw GraphRAGError.config(message: "chunk_overlap must be < chunk_size")
        }
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap
        // For shorter documents the 50-char minimum can drop everything; scale
        // the floor down for small chunk sizes.
        let minSize = min(50, max(1, chunkSize / 4))
        self.chunker = HierarchicalChunker(minChunkSize: minSize)
        self.keywordExtractor = TfIdfKeywordExtractor()
    }

    /// Hierarchically chunk a document into `TextChunk`s with offsets and metadata.
    public func chunk(_ document: Document) -> [TextChunk] {
        let spans = chunker.chunkSpans(document.content, chunkSize: chunkSize, overlap: chunkOverlap)
        var chunks: [TextChunk] = []
        chunks.reserveCapacity(spans.count)
        for (index, span) in spans.enumerated() {
            let id = ChunkID("\(document.id.raw)_\(index)")
            let metadata = ChunkMetadata(
                index: index,
                wordCount: wordCount(span.content),
                keywords: extractKeywords(span.content, maxKeywords: 5)
            )
            chunks.append(
                TextChunk(
                    id: id,
                    documentID: document.id,
                    content: span.content,
                    startOffset: span.startOffset,
                    endOffset: span.endOffset,
                    metadata: metadata
                )
            )
        }
        return chunks
    }

    /// Extract up to `maxKeywords` keywords from `text`.
    public func extractKeywords(_ text: String, maxKeywords: Int) -> [String] {
        keywordExtractor.extractKeywordStrings(text, topK: maxKeywords)
    }

    /// Naive sentence splitter on `.`, `!`, `?`, and newlines.
    public func extractSentences(_ text: String) -> [String] {
        var sentences: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if ch == "." || ch == "!" || ch == "?" || ch == "\n" {
                let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { sentences.append(trimmed) }
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { sentences.append(tail) }
        return sentences
    }

    /// Collapse runs of whitespace and trim.
    public func cleanText(_ text: String) -> String {
        let collapsed = text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        return collapsed.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func wordCount(_ text: String) -> Int {
        text.split(whereSeparator: { $0.isWhitespace }).count
    }
}
