import { ModelType, ChatMessage as ChatMessageType, SourcesResponse } from '../types/chat';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://bde104bopztyao-8000.proxy.runpod.net';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  model_used?: string;
  isCodeResponse?: boolean;
  detectedLanguage?: string;
  timestamp: Date;
}

export interface StreamResponse {
  token: string;
  accumulated_text: string;
  model_used: string;
  model_type: string;
  finished: boolean;
  session_id: string;
  sources?: any[];
  search_query?: string;
  // New source API fields
  source_id?: string;
  source_count?: number;
  search_type?: 'web_search' | 'deep_research';
  language?: string;
  generation_time?: number;
  tokens_generated?: number;
  error?: string;
  // Vision-specific fields
  image_info?: {
    width?: number;
    height?: number;
    file_size_mb?: number;
    file_name?: string;
  };
  config_used?: {
    max_new_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    device?: string;
  };
  capabilities?: string;
  analysis_timestamp?: string;
}

export interface ChatResponse {
  response: string;
  model_used: string;
  tokens_generated: number;
  generation_time: number;
  sources?: any[];
  // New source API fields
  source_id?: string;
  source_count?: number;
  search_type?: 'web_search' | 'deep_research';
}

export interface ModelInfo {
  name: string;
  type: string;
  status: string;
  memory_usage: string;
  performance: string;
}

export class APIService {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  get baseUrl(): string {
    return this.baseURL;
  }

  private async fetchWithError(url: string, options: RequestInit = {}): Promise<Response> {
    const response = await fetch(url, options);
    
    if (!response.ok) {
      const errorText = await response.text();
      let errorMessage;
      
      try {
        const errorJson = JSON.parse(errorText);
        errorMessage = errorJson.detail || errorJson.message || 'Unknown error';
      } catch {
        errorMessage = errorText || `HTTP ${response.status}`;
      }
      
      throw new Error(errorMessage);
    }
    
    return response;
  }

  // Health check
  async healthCheck(): Promise<any> {
    const response = await this.fetchWithError(`${this.baseURL}/health`);
    return response.json();
  }

  // Model management
  async listModels(): Promise<any> {
    const response = await this.fetchWithError(`${this.baseURL}/models`);
    return response.json();
  }

  async loadModel(modelType: string): Promise<any> {
    const response = await this.fetchWithError(`${this.baseURL}/models/${modelType}/load`, {
      method: 'POST',
    });
    return response.json();
  }

  async switchModel(modelType: string): Promise<any> {
    console.log(`🔄 Switching to model: ${modelType}`);
    
    try {
      const response = await Promise.race([
        this.fetchWithError(`${this.baseURL}/models/${modelType}/switch`, {
          method: 'POST',
        }),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Model switch timeout after 10 seconds')), 10000)
        )
      ]);
      
      const result = await (response as Response).json();
      console.log(`✅ Model switch successful:`, result);
      return result;
    } catch (error) {
      console.error(`❌ Model switch failed:`, error);
      throw error;
    }
  }

  async unloadModel(modelType: string): Promise<any> {
    const response = await this.fetchWithError(`${this.baseURL}/models/${modelType}/unload`, {
      method: 'POST',
    });
    return response.json();
  }

  // Chat endpoints with streaming support
  async sendChatMessage(
    message: string,
    sessionId: string = 'default_session',
    options: {
      useWebSearch?: boolean;
      useDeepResearch?: boolean;
      maxTokens?: number;
      temperature?: number;
      modelType?: string;
      onProgress?: (chunk: string) => void;
    } = {}
  ): Promise<any> {
    const requestData = {
      message,
      sessionId: sessionId,
      useWebSearch: options.useWebSearch || false,
      useDeepResearch: options.useDeepResearch || false,
      maxTokens: options.maxTokens || 8000,  // Updated to 8000
      temperature: options.temperature || 0.7,
      modelType: options.modelType || null,  // Include model type if specified
      stream: !!options.onProgress // Enable streaming if callback provided
    };

    const response = await this.fetchWithError(`${this.baseURL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    });

    if (options.onProgress && response.body) {
      return this.handleStreamingResponse(response, options.onProgress);
    }

    return response.json();
  }

  // Streaming chat message generator
  async* sendChatMessageStream(
    message: string,
    sessionId: string = 'default_session',
    options: {
      useWebSearch?: boolean;
      useDeepResearch?: boolean;
      maxTokens?: number;
      temperature?: number;
      modelType?: string;
      files?: File[];
    } = {}
  ): AsyncGenerator<StreamResponse, void, unknown> {
    const requestData = {
      message,
      sessionId: sessionId,
      useWebSearch: options.useWebSearch || false,
      useDeepResearch: options.useDeepResearch || false,
      maxTokens: options.maxTokens || 512,
      temperature: options.temperature || 0.7,
      modelType: options.modelType || null,
      stream: true
    };

    const response = await this.fetchWithError(`${this.baseURL}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    });

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let accumulated = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.token) {
                accumulated += data.token;
                yield {
                  ...data,
                  accumulated_text: accumulated
                };
              }
              if (data.finished) {
                yield {
                  ...data,
                  accumulated_text: accumulated,
                  finished: true
                };
                return;
              }
            } catch (e) {
              // Skip invalid JSON lines
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      throw error;
    }
  }

  // Streaming image analysis
  async* analyzeImageStream(
    image: File,
    prompt: string = 'Describe this image in detail',
    maxTokens: number = 8192,
    useWebSearch: boolean = false,  // Enable web search
    useDeepResearch: boolean = false  // Enable deep research
  ): AsyncGenerator<StreamResponse, void, unknown> {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('prompt', prompt);
    formData.append('max_tokens', maxTokens.toString());
    formData.append('use_web_search', useWebSearch.toString());
    formData.append('use_deep_research', useDeepResearch.toString());
    formData.append('stream', 'true');

    const response = await this.fetchWithError(`${this.baseURL}/vision/analyze/stream`, {
      method: 'POST',
      body: formData,
    });

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let accumulated = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.token) {
                accumulated += data.token;
                yield {
                  ...data,
                  accumulated_text: accumulated
                };
              }
              if (data.finished) {
                yield {
                  ...data,
                  accumulated_text: accumulated,
                  finished: true
                };
                return;
              }
            } catch (e) {
              // Skip invalid JSON lines
            }
          }
        }
      }
    } catch (error) {
      console.error('Image streaming error:', error);
      throw error;
    }
  }

  // Image analysis with Qwen2.5-Omni - Enhanced Settings
  async analyzeImage(
    image: File,
    prompt: string = 'Describe this image in detail',
    maxTokens: number = 8000,  // Updated to 8000 for comprehensive detailed analysis
    useWebSearch: boolean = false,  // Enable web search
    useDeepResearch: boolean = false  // Enable deep research
  ): Promise<any> {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('prompt', prompt);
    formData.append('max_tokens', maxTokens.toString());
    formData.append('use_web_search', useWebSearch.toString());
    formData.append('use_deep_research', useDeepResearch.toString());

    const response = await this.fetchWithError(`${this.baseURL}/vision/analyze`, {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();
    
    // Remove emojis from the response
    if (result.analysis) {
      result.analysis = result.analysis.replace(/[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu, '');
    }
    
    return result;
  }

  // Video analysis with Qwen2.5-Omni (reuse image endpoint for video files)
  async analyzeVideo(
    video: File,
    prompt: string = 'Analyze this video and describe what you see',
    maxTokens: number = 1024
  ): Promise<any> {
    // For now, we can use the same image analysis endpoint since Qwen2.5-Omni handles video frames
    // In a full implementation, you might want a dedicated video endpoint
    const formData = new FormData();
    formData.append('image', video); // The backend can handle video files too
    formData.append('prompt', prompt);
    formData.append('max_tokens', maxTokens.toString());

    const response = await this.fetchWithError(`${this.baseURL}/vision/analyze`, {
      method: 'POST',
      body: formData,
    });

    return response.json();
  }

  // Audio analysis - Qwen2.5-Omni with full audio support
  async analyzeAudio(
    audio: File,
    prompt: string = 'Transcribe and analyze this audio',
    maxTokens: number = 1024
  ): Promise<any> {
    const formData = new FormData();
    formData.append('audio', audio);
    formData.append('prompt', prompt);
    formData.append('max_tokens', maxTokens.toString());

    const response = await this.fetchWithError(`${this.baseURL}/vision/analyze_audio`, {
      method: 'POST',
      body: formData,
    });

    return response.json();
  }

  // Enhanced coding endpoint
  async generateCode(
    prompt: string,
    language: string = 'auto',
    maxTokens: number = 1024,
    temperature: number = 1.0,
    sessionId: string = 'default_session'
  ): Promise<any> {
    const formData = new FormData();
    formData.append('prompt', prompt);
    formData.append('language', language);
    formData.append('max_tokens', maxTokens.toString());
    formData.append('temperature', temperature.toString());
    formData.append('session_id', sessionId);

    const response = await this.fetchWithError(`${this.baseURL}/coding/generate`, {
      method: 'POST',
      body: formData,
    });

    return response.json();
  }

  // Handle streaming responses
  private async handleStreamingResponse(
    response: Response, 
    onProgress: (chunk: string) => void,
    field: string = 'response'
  ): Promise<any> {
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let accumulated = '';
    let finalData: any = {};

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.done) {
                finalData = data;
                break;
              }
              if (data.chunk) {
                accumulated += data.chunk;
                onProgress(accumulated);
              }
            } catch (e) {
              // Skip invalid JSON lines
            }
          }
        }
      }
      
      return {
        [field]: accumulated,
        ...finalData,
        tokens_generated: accumulated.split(/\s+/).length
      };
    } catch (error) {
      console.error('Streaming error:', error);
      throw error;
    }
  }

  // File upload and processing
  async uploadFile(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.fetchWithError(`${this.baseURL}/files/upload`, {
      method: 'POST',
      body: formData,
    });

    return response.json();
  }

  async analyzeFileWithLLM(
    file: File,
    prompt: string = 'Analyze this file and provide insights',
    modelType: string = 'report_generation'
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('prompt', prompt);
    formData.append('model_type', modelType);

    const response = await this.fetchWithError(`${this.baseURL}/files/analyze`, {
      method: 'POST',
      body: formData,
    });

    return response.json();
  }

  // Web search
  async searchWeb(query: string, maxResults: number = 10): Promise<any> {
    const requestData = {
      query,
      max_results: maxResults,
    };

    const response = await this.fetchWithError(`${this.baseURL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    });

    return response.json();
  }

  // Deep research
  async conductResearch(
    query: string,
    depth: number = 3,
    maxSources: number = 10,
    includeAnalysis: boolean = true
  ): Promise<any> {
    const requestData = {
      query,
      depth,
      max_sources: maxSources,
      include_analysis: includeAnalysis,
    };

    const response = await this.fetchWithError(`${this.baseURL}/research`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    });

    return response.json();
  }

  // Report generation
  async generateReport(
    templateName: string,
    data: Record<string, any>,
    outputFormat: string = 'pdf'
  ): Promise<Blob> {
    const requestData = {
      template_name: templateName,
      data,
      output_format: outputFormat,
    };

    const response = await this.fetchWithError(`${this.baseURL}/reports/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    });

    return response.blob();
  }

  async generateResearchReport(
    query: string,
    depth: number = 3,
    outputFormat: string = 'pdf'
  ): Promise<Blob> {
    const requestData = {
      query,
      depth,
      output_format: outputFormat,
    };

    const response = await this.fetchWithError(`${this.baseURL}/reports/generate/research`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    });

    return response.blob();
  }

  async listReportTemplates(): Promise<any> {
    const response = await this.fetchWithError(`${this.baseURL}/reports/templates`);
    return response.json();
  }

  // Embeddings
  async createEmbedding(text: string, instruction?: string): Promise<any> {
    const requestData = {
      text,
      instruction: instruction || 'Represent this text for semantic similarity',
    };

    const response = await this.fetchWithError(`${this.baseURL}/embeddings/create`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    });

    return response.json();
  }

  // Chat history
  async storeChatHistory(messageId: string, sessionId: string, messageType: string, content: string): Promise<any> {
    const requestData = {
      message_id: messageId,
      session_id: sessionId,
      message_type: messageType,
      content,
    };

    const response = await this.fetchWithError(`${this.baseURL}/chat/history/store`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    });

    return response.json();
  }

  async searchChatHistory(query: string, sessionId?: string): Promise<any> {
    const requestData = {
      query,
      session_id: sessionId,
    };

    const response = await this.fetchWithError(`${this.baseURL}/chat/history/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    });

    return response.json();
  }

  // Utility methods
  downloadFile(blob: Blob, filename: string): void {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  }

  // New Sources API methods
  async getSources(sourceId: string): Promise<SourcesResponse> {
    try {
      const response = await this.fetchWithError(`${this.baseURL}/sources/${sourceId}`);
      return response.json();
    } catch (error) {
      console.error('Failed to fetch sources:', error);
      return {
        success: false,
        sources: [],
        count: 0,
        error: error instanceof Error ? error.message : 'Failed to fetch sources'
      };
    }
  }

  async cleanupSources(sourceId: string): Promise<any> {
    try {
      const response = await this.fetchWithError(`${this.baseURL}/sources/${sourceId}`, {
        method: 'DELETE',
      });
      return response.json();
    } catch (error) {
      console.error('Failed to cleanup sources:', error);
      return { status: 'error', message: error instanceof Error ? error.message : 'Cleanup failed' };
    }
  }

  // Advanced Document Processing
  async processDocument(
    file: File,
    modificationPrompt: string,
    outputFormat?: string,
    customFilename?: string
  ): Promise<Blob> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('modification_prompt', modificationPrompt);
    
    if (outputFormat) {
      formData.append('output_format', outputFormat);
    }
    
    if (customFilename) {
      formData.append('custom_filename', customFilename);
    }

    const response = await this.fetchWithError(`${this.baseURL}/documents/process`, {
      method: 'POST',
      body: formData,
    });

    return response.blob();
  }

  async compareDocuments(originalFile: File, modifiedFile: File): Promise<any> {
    const formData = new FormData();
    formData.append('original_file', originalFile);
    formData.append('modified_file', modifiedFile);

    const response = await this.fetchWithError(`${this.baseURL}/documents/compare`, {
      method: 'POST',
      body: formData,
    });

    return response.json();
  }

  async getSupportedDocumentFormats(): Promise<any> {
    const response = await this.fetchWithError(`${this.baseURL}/documents/formats`);
    return response.json();
  }

  async cleanupDocumentFiles(maxAgeHours: number = 24): Promise<any> {
    const response = await this.fetchWithError(`${this.baseURL}/documents/cleanup`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ max_age_hours: maxAgeHours }),
    });

    return response.json();
  }

  // Advanced report generation with progress tracking
  async generateAdvancedReport(request: ReportGenerationRequest) {
    const response = await fetch('/api/reports/generate-advanced', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  // Get report progress
  async getReportProgress(reportId: string) {
    const response = await fetch(`/api/reports/progress/${reportId}`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  // List active reports
  async listActiveReports() {
    const response = await fetch('/api/reports/active');

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  // Cancel report generation
  async cancelReport(reportId: string) {
    const response = await fetch(`/api/reports/cancel/${reportId}`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  // Download completed report
  downloadReport(reportId: string) {
    window.open(`/api/reports/download/${reportId}`, '_blank');
  }
}

// Create and export a default instance
export const apiService = new APIService();

// Export updated types for TypeScript support
export interface ChatRequest {
  message: string;
  session_id: string;
  use_web_search?: boolean;
  use_deep_research?: boolean;
  max_tokens?: number;
  temperature?: number;
}

export interface FileInfo {
  filename: string;
  size: number;
  extension: string;
  mime_type: string;
  category: string;
  content?: string;
  temp_path?: string;
}

export interface ResearchResult {
  query: string;
  research_results: {
    summary: string;
    key_points: string[];
    contradictions: any[];
    analysis: string;
    sources: Array<{
      title: string;
      url: string;
      content: string;
    }>;
    confidence_score: number;
  };
  timestamp: string;
}

// Streaming chat function
export async function* sendMessageStream(
  message: string,
  sessionId: string = 'default_session',
  useWebSearch: boolean = false,
  useDeepResearch: boolean = false,
  maxTokens: number = 8000,  // Updated to 8000
  temperature: number = 0.7
): AsyncGenerator<StreamResponse, void, unknown> {
  console.log('sendMessageStream called with:', { message, sessionId, useWebSearch, useDeepResearch, maxTokens, temperature });
  
  const requestBody = {
    message: message,
    sessionId: sessionId,
    useWebSearch: useWebSearch,
    useDeepResearch: useDeepResearch,
    maxTokens: maxTokens,
    temperature: temperature
  };

  console.log('Making fetch request to:', `${API_BASE_URL}/chat/stream`);
  console.log('Request body:', requestBody);

  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  });

  console.log('Response status:', response.status);
  console.log('Response headers:', response.headers);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('Stream not available');
  }

  const decoder = new TextDecoder();
  let buffer = '';
  let chunkCount = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log('Stream finished, total chunks processed:', chunkCount);
        break;
      }

      const chunk = decoder.decode(value, { stream: true });
      console.log(`Raw chunk ${chunkCount}:`, chunk);
      
      buffer += chunk;
      
      // Process complete lines
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        console.log('Processing line:', line);
        if (line.trim() && line.startsWith('data: ')) {
          try {
            const jsonStr = line.slice(6);
            console.log('Parsing JSON:', jsonStr);
            const data = JSON.parse(jsonStr);
            console.log('Parsed data:', data);
            chunkCount++;
            yield data as StreamResponse;
          } catch (e) {
            console.error('Error parsing stream data:', e);
            console.error('Problematic line:', line);
          }
        } else if (line.trim() && !line.startsWith('data: ')) {
          console.log('Non-data line (ignored):', line);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

// Streaming vision analysis function
export async function* analyzeImageStream(
  image: File,
  prompt: string = 'Analyze this image in detail',
  maxTokens: number = 8192,
  useWebSearch: boolean = false,  // Enable web search
  useDeepResearch: boolean = false  // Enable deep research
): AsyncGenerator<StreamResponse, void, unknown> {
  console.log('analyzeImageStream called with:', { fileName: image.name, prompt, maxTokens, useWebSearch, useDeepResearch });
  
  const formData = new FormData();
  formData.append('image', image);
  formData.append('prompt', prompt);
  formData.append('max_tokens', maxTokens.toString());
  formData.append('use_web_search', useWebSearch.toString());
  formData.append('use_deep_research', useDeepResearch.toString());

  console.log('Making fetch request to:', `${API_BASE_URL}/vision/analyze/stream`);

  const response = await fetch(`${API_BASE_URL}/vision/analyze/stream`, {
    method: 'POST',
    body: formData,
  });

  console.log('Vision stream response status:', response.status);
  console.log('Vision stream response headers:', response.headers);

  if (!response.ok) {
    const errorText = await response.text();
    console.error('Vision stream error response:', errorText);
    throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('Vision stream not available');
  }

  const decoder = new TextDecoder();
  let buffer = '';
  let chunkCount = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log('Vision stream finished, total chunks processed:', chunkCount);
        break;
      }

      const chunk = decoder.decode(value, { stream: true });
      console.log(`Raw vision chunk ${chunkCount}:`, chunk);
      
      buffer += chunk;
      
      // Process complete lines
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        console.log('Processing vision line:', line);
        if (line.trim() && line.startsWith('data: ')) {
          try {
            const jsonStr = line.slice(6);
            console.log('Parsing vision JSON:', jsonStr);
            const data = JSON.parse(jsonStr);
            console.log('Parsed vision data:', data);
            chunkCount++;
            
            // Extend the data with vision-specific fields
            const visionData = {
              ...data,
              token: data.token || '',
              accumulated_text: data.accumulated_text || '',
              model_used: data.model_used || 'qwen2.5-vl',
              model_type: data.model_type || 'qwen2.5-vl',
              finished: data.finished || false,
              session_id: data.session_id || 'vision_session',
              image_info: data.image_info || {},
              config_used: data.config_used || {},
              capabilities: data.capabilities || 'Advanced vision analysis',
              analysis_timestamp: data.analysis_timestamp
            };
            
            yield visionData as StreamResponse;
          } catch (e) {
            console.error('Error parsing vision stream data:', e);
            console.error('Problematic vision line:', line);
          }
        } else if (line.trim() && !line.startsWith('data: ')) {
          console.log('Non-data vision line (ignored):', line);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

// Streaming code generation function
export async function* generateCodeStream(
  prompt: string,
  language: string = 'auto',
  maxTokens: number = 1024,
  temperature: number = 1.0,
  sessionId: string = 'default_session'
): AsyncGenerator<StreamResponse, void, unknown> {
  console.log('generateCodeStream called with:', { prompt, language, maxTokens, temperature, sessionId });
  
  const formData = new FormData();
  formData.append('prompt', prompt);
  formData.append('language', language);
  formData.append('max_tokens', maxTokens.toString());
  formData.append('temperature', temperature.toString());
  formData.append('session_id', sessionId);

  console.log('Making fetch request to:', `${API_BASE_URL}/coding/generate/stream`);

  const response = await fetch(`${API_BASE_URL}/coding/generate/stream`, {
    method: 'POST',
    body: formData,
  });

  console.log('Response status:', response.status);
  console.log('Response headers:', response.headers);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('Stream not available');
  }

  const decoder = new TextDecoder();
  let buffer = '';
  let chunkCount = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log('Code stream finished, total chunks processed:', chunkCount);
        break;
      }

      const chunk = decoder.decode(value, { stream: true });
      console.log(`Raw code chunk ${chunkCount}:`, chunk);
      
      buffer += chunk;
      
      // Process complete lines
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        console.log('Processing code line:', line);
        if (line.trim() && line.startsWith('data: ')) {
          try {
            const jsonStr = line.slice(6);
            console.log('Parsing code JSON:', jsonStr);
            const data = JSON.parse(jsonStr);
            console.log('Parsed code data:', data);
            chunkCount++;
            yield data as StreamResponse;
          } catch (e) {
            console.error('Error parsing code stream data:', e);
            console.error('Problematic code line:', line);
          }
        } else if (line.trim() && !line.startsWith('data: ')) {
          console.log('Non-data code line (ignored):', line);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

interface ReportGenerationRequest {
  content: string;
  report_type: string;
  output_format: string;
  custom_requirements?: string;
  enable_advanced_research: boolean;
  enable_complex_charts: boolean;
  enable_deep_analysis: boolean;
  chart_theme: string;
} 