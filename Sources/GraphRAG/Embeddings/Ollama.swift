// Ollama.swift
// Ported from graphrag-rs `ollama` and `embeddings::ollama`.
//
// Talks to a local Ollama daemon over HTTP: `/api/generate` for completions and
// `/api/embeddings` for embeddings. Network calls go through URLSession.

import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// Connection + generation settings for a local Ollama server.
public struct OllamaConfig: Sendable {
    public var host: String
    public var port: Int
    public var chatModel: String
    public var embeddingModel: String
    public var embeddingDimension: Int
    public var temperature: Float
    public var maxTokens: Int
    public var timeoutSeconds: Double
    public var keepAlive: String?
    public var numCtx: Int?

    public init(
        host: String = "http://localhost",
        port: Int = 11434,
        chatModel: String = "llama3.2:3b",
        embeddingModel: String = "nomic-embed-text",
        embeddingDimension: Int = 1024,
        temperature: Float = 0.7,
        maxTokens: Int = 2000,
        timeoutSeconds: Double = 30,
        keepAlive: String? = nil,
        numCtx: Int? = nil
    ) {
        self.host = host
        self.port = port
        self.chatModel = chatModel
        self.embeddingModel = embeddingModel
        self.embeddingDimension = embeddingDimension
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.timeoutSeconds = timeoutSeconds
        self.keepAlive = keepAlive
        self.numCtx = numCtx
    }

    var baseURL: String {
        // Accept bare hosts ("localhost", "127.0.0.1"): without a scheme, URL
        // parses the host as the scheme and the request fails.
        let normalizedHost = host.contains("://") ? host : "http://\(host)"
        // If the host already includes a port (e.g. "http://localhost:11434"),
        // don't append another.
        if let schemeRange = normalizedHost.range(of: "://"),
            normalizedHost[schemeRange.upperBound...].contains(":")
        {
            return normalizedHost
        }
        return "\(normalizedHost):\(port)"
    }
}

/// Shared low-level HTTP helpers for the Ollama REST API.
enum OllamaHTTP {
    /// Serialize a JSON object to `Data` (synchronous; nothing crosses an await).
    static func encode(_ body: [String: Any]) throws -> Data {
        do {
            return try JSONSerialization.data(withJSONObject: body)
        } catch {
            throw GraphRAGError.serialization(message: error.localizedDescription)
        }
    }

    static func post(
        urlString: String, jsonBody: Data, timeout: Double
    ) async throws -> Data {
        guard let url = URL(string: urlString) else {
            throw GraphRAGError.network(message: "Invalid Ollama URL: \(urlString)")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonBody
        return try await perform(request)
    }

    static func get(urlString: String, timeout: Double) async throws -> Data {
        guard let url = URL(string: urlString) else {
            throw GraphRAGError.network(message: "Invalid Ollama URL: \(urlString)")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = timeout
        return try await perform(request)
    }

    private static func perform(_ request: URLRequest) async throws -> Data {
        // Async-native URLSession supports task cancellation, unlike the legacy
        // callback API wrapped in a continuation.
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw GraphRAGError.network(message: error.localizedDescription)
        }
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw GraphRAGError.http(message: "HTTP \(http.statusCode)")
        }
        return data
    }
}

/// `LanguageModel` backed by Ollama's `/api/generate`.
public struct OllamaClient: LanguageModel {
    public let config: OllamaConfig

    public init(config: OllamaConfig = OllamaConfig()) {
        self.config = config
    }

    public var modelInfo: ModelInfo {
        ModelInfo(
            name: config.chatModel, maxContextLength: config.numCtx, supportsStreaming: true)
    }

    public func isAvailable() async -> Bool {
        do {
            _ = try await OllamaHTTP.get(
                urlString: "\(config.baseURL)/api/tags", timeout: config.timeoutSeconds)
            return true
        } catch {
            return false
        }
    }

    public func complete(_ prompt: String, params: GenerationParams) async throws -> String {
        var options: [String: Any] = [
            "temperature": Double(params.temperature ?? config.temperature),
            "num_predict": params.maxTokens ?? config.maxTokens,
        ]
        if let topP = params.topP { options["top_p"] = Double(topP) }
        if let numCtx = config.numCtx { options["num_ctx"] = numCtx }
        if let stop = params.stopSequences { options["stop"] = stop }

        var body: [String: Any] = [
            "model": config.chatModel,
            "prompt": prompt,
            "stream": false,
            "options": options,
        ]
        if let keepAlive = config.keepAlive { body["keep_alive"] = keepAlive }

        let jsonBody = try OllamaHTTP.encode(body)
        let data = try await OllamaHTTP.post(
            urlString: "\(config.baseURL)/api/generate", jsonBody: jsonBody,
            timeout: config.timeoutSeconds)
        struct GenerateResponse: Codable { let response: String }
        do {
            return try JSONDecoder().decode(GenerateResponse.self, from: data).response
        } catch {
            throw GraphRAGError.generation(message: "Failed to decode Ollama response")
        }
    }
}

/// `EmbeddingModel` backed by Ollama's `/api/embeddings`.
public struct OllamaEmbedder: EmbeddingModel {
    public let config: OllamaConfig

    public init(config: OllamaConfig = OllamaConfig()) {
        self.config = config
    }

    public var dimension: Int { config.embeddingDimension }

    public func isAvailable() async -> Bool {
        do {
            _ = try await OllamaHTTP.get(
                urlString: "\(config.baseURL)/api/tags", timeout: config.timeoutSeconds)
            return true
        } catch {
            return false
        }
    }

    public func embed(_ text: String) async throws -> [Float] {
        let body: [String: Any] = ["model": config.embeddingModel, "prompt": text]
        let jsonBody = try OllamaHTTP.encode(body)
        let data = try await OllamaHTTP.post(
            urlString: "\(config.baseURL)/api/embeddings", jsonBody: jsonBody,
            timeout: config.timeoutSeconds)
        struct EmbeddingResponse: Codable { let embedding: [Float] }
        do {
            return try JSONDecoder().decode(EmbeddingResponse.self, from: data).embedding
        } catch {
            throw GraphRAGError.embedding(message: "Failed to decode Ollama embedding")
        }
    }
}
