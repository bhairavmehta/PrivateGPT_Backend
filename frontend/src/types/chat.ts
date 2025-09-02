export type ModelType = 'chat' | 'coding' | 'vision' | 'report_generation';

export interface ChatMessage {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  attachments?: File[];
  isCodeResponse?: boolean;
  sources?: any[];
  search_query?: string;
  // New source API fields
  source_id?: string;
  source_count?: number;
  search_type?: 'web_search' | 'deep_research';
}

export interface ChatSession {
  id: string;
  name: string;
  messages: ChatMessage[];
  settings: ChatSettings;
  createdAt: Date;
  updatedAt: Date;
}

export interface ChatSettings {
  model: ModelType;
  deepResearch: boolean;
  webSearch: boolean;
}

export interface Attachment {
  id: string;
  name: string;
  type: string;
  size: number;
  url?: string;
  base64?: string;
}

export interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  attachments?: Attachment[];
  model?: string;
  isStreaming?: boolean;
  isReportGeneration?: boolean;
  reportId?: string;
  reportProgress?: ReportProgress;
  // New source API fields
  source_id?: string;
  source_count?: number;
  search_type?: 'web_search' | 'deep_research';
  sources?: any[];
  search_query?: string;
}

// Sources API response interface
export interface SourcesResponse {
  success: boolean;
  sources: Array<{
    title: string;
    url: string;
    content: string;
    score?: number;
  }>;
  search_query?: string;
  timestamp?: number;
  type?: 'web_search' | 'deep_research';
  count: number;
  error?: string;
}

// Progress tracking interfaces
export interface ProgressPhase {
  name: string;
  description: string;
  total_steps: number;
  current_step: number;
  status: 'pending' | 'running' | 'completed' | 'error';
  start_time?: string;
  end_time?: string;
  error_message?: string;
}

export interface ReportProgress {
  report_id: string;
  status: 'queued' | 'running' | 'completed' | 'error' | 'cancelled';
  overall_progress: number;
  current_phase: string;
  phases: ProgressPhase[];
  start_time?: string;
  end_time?: string;
  estimated_completion?: string;
  model_calls_made: number;
  total_model_calls_planned: number;
  result?: any;
  error_message?: string;
  elapsed_time?: number;
  estimated_remaining_seconds?: number;
}
