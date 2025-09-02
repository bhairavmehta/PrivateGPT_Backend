import React, { useState, useRef, useEffect } from 'react';
import { Send, Plus, Paperclip, Globe, Search, Mic, X } from 'lucide-react';
import VoiceRecorder from './VoiceRecorder';
import WebSearch from './WebSearch';
import FileUpload from './FileUpload';
import TemplateUpload from './TemplateUpload';
import { ModelType } from '@/types/chat';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { toast } from "@/components/ui/use-toast";

interface FloatingChatInputProps {
  onSendMessage: (message: string, files?: File[]) => void;
  disabled?: boolean;
  deepResearch?: boolean;
  onToggleDeepResearch?: () => void;
  webSearch?: boolean;
  onToggleWebSearch?: () => void;
  currentModel?: ModelType;
}

const FloatingChatInput: React.FC<FloatingChatInputProps> = ({
  onSendMessage,
  disabled,
  deepResearch,
  onToggleDeepResearch,
  webSearch,
  onToggleWebSearch,
  currentModel
}) => {
  const [message, setMessage] = useState('');
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [activeTools, setActiveTools] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const templateInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if ((message.trim() || attachedFiles.length > 0) && !disabled) {
      const messageToSend = message.trim();
      const filesToSend = [...attachedFiles];
      
      // Clear input immediately to prevent contamination
      setMessage('');
      setAttachedFiles([]);
      setActiveTools([]);
      
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
      
      // Send the message
      onSendMessage(messageToSend, filesToSend);
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

  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleTemplateUpload = () => {
    templateInputRef.current?.click();
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files);
      const uniqueNewFiles = newFiles.filter(
        (file) => !attachedFiles.some((att) => att.name === file.name && att.size === file.size)
      );

      if (uniqueNewFiles.length !== newFiles.length) {
        toast({
          title: 'Duplicate Files Skipped',
          description: 'Some of the selected files were already attached.',
        });
      }

      setAttachedFiles((prev) => [...prev, ...uniqueNewFiles]);
    }
    // Correctly reset the file input to allow re-uploading the same file
    if(e.target) {
      e.target.value = '';
    }
  };

  // Get appropriate file types based on current model
  const getAcceptedFileTypes = () => {
    switch (currentModel) {
      case 'coding':
        // Code files and documents only (no multimedia)
        return '.pdf,.doc,.docx,.txt,.md,.rtf,.js,.jsx,.ts,.tsx,.py,.java,.cpp,.c,.h,.hpp,.cs,.php,.rb,.go,.rs,.kt,.swift,.scala,.r,.sql,.html,.css,.scss,.sass,.less,.xml,.json,.yaml,.yml,.toml,.ini,.cfg,.conf,.rst,.sh,.bash,.zsh,.fish,.ps1,.bat,.cmd,.dockerfile,.makefile,.gradle,.maven,.cmake,.vue,.svelte,.dart,.lua,.perl,.pl,.clj,.cljs,.edn,.elm,.ex,.exs,.fs,.fsx,.hs,.lhs,.ml,.mli,.nim,.rkt,.scm,.ss,.lisp,.cl,.jl,.m,.mm,.pas,.pp,.inc,.asm,.s,.vb,.vbs,.vbnet,.f,.f90,.f95,.for,.ftn,.fpp,.ada,.adb,.ads,.cob,.cbl,.cobol,.pro,.prolog,.tcl,.tk,.awk,.sed,.vim,.emacs,.gitignore,.env';
      
      case 'vision':
        // Full multimodal support - Qwen2.5-Omni supports images, videos, audio, and documents
        return '.png,.jpg,.jpeg,.gif,.webp,.svg,.bmp,.ico,.pdf,.doc,.docx,.txt,.md,.rtf,.mp4,.mov,.avi,.webm,.mkv,.flv,.wmv,.m4v,.mp3,.wav,.ogg,.m4a,.aac,.flac,.wma,.opus';
      
      default:
        // Report generation: all file types including multimedia
        return '.pdf,.doc,.docx,.txt,.md,.rtf,.png,.jpg,.jpeg,.gif,.webp,.svg,.bmp,.ico,.mp4,.mov,.avi,.webm,.mkv,.mp3,.wav,.ogg,.m4a,.aac,.flac,.wma,.opus,.js,.jsx,.ts,.tsx,.py,.java,.cpp,.c,.h,.hpp,.cs,.php,.rb,.go,.rs,.kt,.swift,.scala,.r,.sql,.html,.css,.scss,.sass,.less,.xml,.json,.yaml,.yml,.toml,.ini,.cfg,.conf,.rst,.sh,.bash,.zsh,.fish,.ps1,.bat,.cmd,.dockerfile,.makefile,.gradle,.maven,.cmake,.vue,.svelte,.dart,.lua,.perl,.pl,.clj,.cljs,.edn,.elm,.ex,.exs,.fs,.fsx,.hs,.lhs,.ml,.mli,.nim,.rkt,.scm,.ss,.lisp,.cl,.jl,.m,.mm,.pas,.pp,.inc,.asm,.s,.vb,.vbs,.vbnet,.f,.f90,.f95,.for,.ftn,.fpp,.ada,.adb,.ads,.cob,.cbl,.cobol,.pro,.prolog,.tcl,.tk,.awk,.sed,.vim,.emacs,.gitignore,.env';
    }
  };

  const getFileTypeLabel = () => {
    switch (currentModel) {
      case 'coding':
        return 'Upload code files and documents';
      case 'vision':
        return 'Upload images, videos, audio, and documents';
      default:
        return 'Upload any file type for analysis';
    }
  };

  const toggleTool = (toolName: string) => {
    setActiveTools(prev => 
      prev.includes(toolName) 
        ? prev.filter(tool => tool !== toolName)
        : [...prev, toolName]
    );
  };

  const handleDeepResearchToggle = () => {
    onToggleDeepResearch?.();
  };

  const handleWebSearchToggle = () => {
    onToggleWebSearch?.();
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as React.FormEvent);
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [message]);

  return (
    <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 w-full max-w-4xl px-4 z-50 lg:pl-64">
      <div className="bg-card backdrop-blur-xl rounded-2xl shadow-2xl border border-border">
        
        {/* Active Tools Display */}
        {(activeTools.length > 0 || attachedFiles.length > 0 || deepResearch || webSearch) && (
          <div className="px-4 pt-4 pb-2">
            <div className="flex flex-wrap gap-2">
              {deepResearch && (
                <div className="flex items-center gap-2 bg-secondary text-secondary-foreground px-3 py-2 rounded-full text-sm font-medium">
                  <div className="w-2 h-2 bg-muted-foreground rounded-full"></div>
                  <span>Deep Research</span>
                  <button
                    onClick={handleDeepResearchToggle}
                    className="ml-1 hover:bg-accent rounded-full p-1 transition-colors"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              )}
              {webSearch && (
                <div className="flex items-center gap-2 bg-secondary text-secondary-foreground px-3 py-2 rounded-full text-sm font-medium">
                  <div className="w-2 h-2 bg-muted-foreground rounded-full"></div>
                  <span>Web Search</span>
                  <button
                    onClick={handleWebSearchToggle}
                    className="ml-1 hover:bg-accent rounded-full p-1 transition-colors"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              )}
              {activeTools.map((tool, index) => (
                <div key={index} className="flex items-center gap-2 bg-secondary text-secondary-foreground px-3 py-2 rounded-full text-sm font-medium">
                  <div className="w-2 h-2 bg-muted-foreground rounded-full"></div>
                  <span>{tool}</span>
                  <button
                    onClick={() => toggleTool(tool)}
                    className="ml-1 hover:bg-accent rounded-full p-1 transition-colors"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
              {attachedFiles.map((file, index) => {
                const isTemplate = currentModel === 'report_generation' && ['.pdf', '.doc', '.docx', '.txt', '.md'].includes('.' + file.name.split('.').pop()?.toLowerCase());
                return (
                  <div key={index} className={`flex items-center gap-2 px-3 py-2 rounded-full text-sm font-medium ${
                    isTemplate ? 'bg-orange-100 text-orange-800 border border-orange-200' : 'bg-secondary text-secondary-foreground'
                  }`}>
                    {isTemplate ? (
                      <span className="text-orange-600">📄</span>
                    ) : (
                  <Paperclip className="w-3 h-3" />
                    )}
                  <span>{file.name.length > 20 ? file.name.substring(0, 20) + '...' : file.name}</span>
                    {isTemplate && <span className="text-xs bg-orange-200 px-1 rounded">TEMPLATE</span>}
                  <button
                    onClick={() => removeFile(index)}
                    className="ml-1 hover:bg-accent rounded-full p-1 transition-colors"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Input Area */}
        <form onSubmit={handleSubmit} className="relative">
          <div className="flex items-center gap-3 p-4">
            {/* Tools Dropdown */}
            <div className="flex-shrink-0">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button
                    type="button"
                    className="p-3 text-muted-foreground hover:text-foreground hover:bg-accent rounded-xl transition-colors flex items-center justify-center"
                    aria-label="Tools"
                  >
                    <Plus className="w-5 h-5" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-72 bg-popover border border-border shadow-2xl z-[60] rounded-xl">
                  <DropdownMenuItem onClick={handleFileUpload} className="flex items-center gap-3 p-4 hover:bg-accent rounded-lg m-1 transition-colors cursor-pointer">
                    <div className="p-2 bg-secondary rounded-lg">
                      <Paperclip className="w-4 h-4 text-secondary-foreground" />
                    </div>
                    <div>
                      <div className="font-semibold text-popover-foreground">Add files</div>
                      <div className="text-xs text-muted-foreground">{getFileTypeLabel()}</div>
                    </div>
                  </DropdownMenuItem>
                  
                  {currentModel === 'report_generation' && (
                    <DropdownMenuItem onClick={handleTemplateUpload} className="flex items-center gap-3 p-4 hover:bg-accent rounded-lg m-1 transition-colors cursor-pointer">
                      <div className="p-2 bg-orange-100 rounded-lg">
                        <Paperclip className="w-4 h-4 text-orange-600" />
                      </div>
                      <div>
                        <div className="font-semibold text-popover-foreground">📄 Add template</div>
                        <div className="text-xs text-muted-foreground">Upload PDF/DOCX template to guide report structure</div>
                      </div>
                    </DropdownMenuItem>
                  )}
                  
                  <DropdownMenuSeparator className="my-2 bg-border" />
                  
                  <DropdownMenuItem 
                    onClick={handleDeepResearchToggle}
                    className="flex items-center gap-3 p-4 hover:bg-accent rounded-lg m-1 transition-colors cursor-pointer"
                  >
                    <div className="p-2 bg-secondary rounded-lg">
                      <Search className="w-4 h-4 text-secondary-foreground" />
                    </div>
                    <div>
                      <div className="font-semibold text-popover-foreground">Deep research</div>
                      <div className="text-xs text-muted-foreground">Enhanced analysis and insights</div>
                    </div>
                  </DropdownMenuItem>
                  
                  <DropdownMenuItem 
                    onClick={handleWebSearchToggle}
                    className="flex items-center gap-3 p-4 hover:bg-accent rounded-lg m-1 transition-colors cursor-pointer"
                  >
                    <div className="p-2 bg-secondary rounded-lg">
                      <Globe className="w-4 h-4 text-secondary-foreground" />
                    </div>
                    <div>
                      <div className="font-semibold text-popover-foreground">Web search</div>
                      <div className="text-xs text-muted-foreground">Search the internet for information</div>
                    </div>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            {/* Text Input */}
            <div className="flex-1 relative min-h-[56px]">
              <textarea
                ref={textareaRef}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder={
                  currentModel === 'coding' 
                    ? "Ask for code in any language (Python, JavaScript, etc.)..."
                    : currentModel === 'vision'
                    ? "Analyze images, videos, audio, or documents with Qwen2.5-Omni..."
                    : "Message AI Assistant..."
                }
                className="w-full p-4 pr-16 border-0 bg-input text-foreground placeholder-muted-foreground focus:ring-1 focus:ring-ring focus:outline-none resize-none rounded-xl transition-colors min-h-[56px] max-h-[120px]"
                rows={1}
                disabled={disabled}
              />
              
              {/* Voice Recorder */}
              <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
                <VoiceRecorder onTranscription={handleVoiceTranscription} disabled={disabled} />
              </div>
            </div>

            {/* Send Button */}
            <div className="flex-shrink-0">
              <button
                type="submit"
                disabled={disabled || (!message.trim() && attachedFiles.length === 0)}
                className="p-4 bg-secondary text-secondary-foreground rounded-xl hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center min-h-[56px]"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </form>

        {/* Hidden File Inputs */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          onChange={onFileChange}
          className="hidden"
          accept={getAcceptedFileTypes()}
        />
        {currentModel === 'report_generation' && (
          <input
            ref={templateInputRef}
            type="file"
            multiple={false}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                const templateFile = e.target.files[0];
                const allowedTypes = ['.pdf', '.doc', '.docx', '.txt', '.md'];
                const fileExtension = '.' + templateFile.name.split('.').pop()?.toLowerCase();
                
                if (allowedTypes.includes(fileExtension)) {
                  setAttachedFiles(prev => [...prev, templateFile]);
                  toast({
                    title: "Template Added",
                    description: `Template "${templateFile.name}" will guide report structure.`,
                  });
                } else {
                  toast({
                    title: "Invalid Template",
                    description: "Templates must be PDF, DOC, DOCX, TXT, or MD files.",
                    variant: "destructive"
                  });
                }
              }
              if (e.target) {
                e.target.value = '';
              }
            }}
            className="hidden"
            accept=".pdf,.doc,.docx,.txt,.md"
          />
        )}
      </div>
    </div>
  );
};

export default FloatingChatInput;
