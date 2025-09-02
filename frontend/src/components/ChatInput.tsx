
import React, { useState } from 'react';
import { Send } from 'lucide-react';
import FileUpload from './FileUpload';
import TemplateUpload from './TemplateUpload';
import VoiceRecorder from './VoiceRecorder';
import WebSearch from './WebSearch';
import { ModelType } from '@/types/chat';

interface ChatInputProps {
  onSendMessage: (content: string, files?: File[]) => void;
  disabled?: boolean;
  deepResearch: boolean;
  onToggleDeepResearch: () => void;
  currentModel: ModelType;
}

const ChatInput: React.FC<ChatInputProps> = ({ 
  onSendMessage, 
  disabled, 
  deepResearch, 
  onToggleDeepResearch,
  currentModel 
}) => {
  const [message, setMessage] = useState('');
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() || attachedFiles.length > 0) {
      onSendMessage(message.trim(), attachedFiles);
      setMessage('');
      setAttachedFiles([]);
    }
  };

  const handleFilesSelected = (files: File[]) => {
    setAttachedFiles(prev => [...prev, ...files]);
  };

  const removeFile = (index: number) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleVoiceTranscription = (transcript: string) => {
    setMessage(prev => prev + (prev ? ' ' : '') + transcript);
  };

  const handleWebSearchResults = (results: string) => {
    setMessage(prev => prev + (prev ? '\n\n' : '') + results);
  };

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4">
      {/* Deep Research Toggle - Available for all models */}
      <div className="mb-3">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={deepResearch}
            onChange={onToggleDeepResearch}
            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Enable deep research mode
          </span>
        </label>
      </div>

      {/* Attached Files */}
      {attachedFiles.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {attachedFiles.map((file, index) => (
            <div key={index} className="flex items-center gap-1 bg-blue-100 dark:bg-blue-900 px-2 py-1 rounded-md text-xs">
              <span>{file.name}</span>
              <button
                onClick={() => removeFile(index)}
                className="text-red-500 hover:text-red-700 ml-1"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <form onSubmit={handleSubmit} className="flex items-end gap-2">
        <div className="flex-1 relative">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type your message or use voice recording..."
            className="w-full p-3 pr-32 border border-gray-300 dark:border-gray-600 rounded-xl bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={1}
            style={{ minHeight: '48px', maxHeight: '120px' }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
          />
          <div className="absolute right-2 bottom-2 flex items-center gap-1">
            {/* Voice recording - available for all models */}
            <VoiceRecorder onTranscription={handleVoiceTranscription} disabled={disabled} />
            
            {/* Web search - available for all models */}
            <WebSearch onSearchResults={handleWebSearchResults} disabled={disabled} />
            
            {/* General file upload - available for all models */}
            <FileUpload onFilesSelected={handleFilesSelected} disabled={disabled} />
            
            {/* Template upload - only for report generation model */}
            {currentModel === 'report' && (
              <TemplateUpload onFilesSelected={handleFilesSelected} disabled={disabled} />
            )}
          </div>
        </div>
        <button
          type="submit"
          disabled={disabled || (!message.trim() && attachedFiles.length === 0)}
          className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg"
        >
          <Send className="w-5 h-5" />
        </button>
      </form>
    </div>
  );
};

export default ChatInput;
