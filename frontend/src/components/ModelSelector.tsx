import React, { useState } from 'react';
import { ModelType } from '@/types/chat';
import { Eye, FileText, Code, ChevronDown, Loader2, DownloadCloud } from 'lucide-react';
import * as SelectPrimitive from "@radix-ui/react-select";
import { apiService } from '@/services/api';
import { useToast } from '@/hooks/use-toast';

interface ModelSelectorProps {
  selectedModel: ModelType;
  onModelChange: (model: ModelType) => void;
}

const models: { id: ModelType | string; name: string; description: string; icon: React.ReactNode; isDownload?: boolean; downloadUrl?: string; }[] = [
  {
    id: 'report_generation',
    name: 'Report Generation (Lite)',
    description: 'For detailed reports and analysis',
    icon: <FileText className="w-4 h-4" />,
  },
  {
    id: 'vision',
    name: 'Vision Model (Lite)',
    description: 'For advanced image, video, and visual analysis',
    icon: <Eye className="w-4 h-4" />,
  },
  {
    id: 'coding',
    name: 'Coding Assistant (Lite)',
    description: 'For advanced programming and code analysis',
    icon: <Code className="w-4 h-4" />,
  },
  {
    id: 'report_generation_pro',
    name: 'Report Generation',
    description: 'Download the full-power model for report generation.',
    icon: <DownloadCloud className="w-4 h-4 text-green-500" />,
    isDownload: true,
    downloadUrl: 'https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/qwen2.5-coder-32b-instruct-q8_0.gguf',
  },
  {
    id: 'vision_pro',
    name: 'Vision Model',
    description: 'Download the full-power model for vision analysis.',
    icon: <DownloadCloud className="w-4 h-4 text-green-500" />,
    isDownload: true,
    downloadUrl: 'https://huggingface.co/unsloth/Qwen2.5-VL-72B-Instruct-GGUF/resolve/main/Qwen2.5-VL-72B-Instruct-Q5_K_XL.gguf',
  },
  {
    id: 'coding_pro',
    name: 'Coding Assistant',
    description: 'Download the full-power model for coding.',
    icon: <DownloadCloud className="w-4 h-4 text-green-500" />,
    isDownload: true,
    downloadUrl: 'https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/qwen2.5-coder-32b-instruct-q8_0.gguf',
  },
];

const ModelSelector: React.FC<ModelSelectorProps> = ({ selectedModel, onModelChange }) => {
  const [switching, setSwitching] = useState(false);
  const { toast } = useToast();
  const currentModel = models.find((m) => m.id === selectedModel) || models[0];

  const handleModelChange = async (newModelId: string) => {
    const model = models.find(m => m.id === newModelId);
    if (!model) return;

    if (model.isDownload && model.downloadUrl) {
      window.open(model.downloadUrl, '_blank');
      toast({
        title: "Download Started",
        description: `Your download for ${model.name} has started.`,
      });
      return;
    }
    
    if (newModelId === selectedModel) return;
    
    setSwitching(true);
    console.log(`🔄 Starting model switch to: ${newModelId}`);
    
    try {
      const result = await apiService.switchModel(newModelId as ModelType);
      console.log(`✅ Model switch result:`, result);
      
      onModelChange(newModelId as ModelType);
      
      toast({
        title: "Model Switched",
        description: `Successfully switched to ${model.name}`,
      });
    } catch (error) {
      console.error('❌ Error switching model:', error);
      
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : 'Failed to switch model',
        variant: "destructive",
      });
    } finally {
      setSwitching(false);
    }
  };

  return (
    <div className="relative">
      <SelectPrimitive.Root value={selectedModel} onValueChange={handleModelChange} disabled={switching}>
        <SelectPrimitive.Trigger className="w-auto min-w-[160px] h-8 bg-transparent border-none hover:bg-muted transition-colors rounded-lg px-2 py-1 text-sm focus:ring-0 focus:ring-offset-0 flex items-center justify-between">
          <SelectPrimitive.Value>
            <div className="flex items-center gap-2">
              {switching ? (
                <Loader2 className="w-4 h-4 text-muted-foreground animate-spin" />
              ) : (
                currentModel.icon
              )}
              <span className="text-sm text-muted-foreground font-normal">
                {switching ? 'Switching...' : currentModel.name}
              </span>
            </div>
          </SelectPrimitive.Value>
          <SelectPrimitive.Icon asChild>
            <ChevronDown className="h-4 w-4 opacity-50" />
          </SelectPrimitive.Icon>
        </SelectPrimitive.Trigger>
        
        <SelectPrimitive.Portal>
          <SelectPrimitive.Content className="min-w-[380px] max-w-[400px] bg-popover border border-border shadow-2xl rounded-xl p-2 z-50 relative max-h-96 overflow-y-auto data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95">
            <SelectPrimitive.Viewport>
              <div className="mb-3 px-3 py-2">
                <div className="flex items-center gap-2 text-muted-foreground text-sm">
                  <span>AI Models</span>
                  <div className="w-4 h-4 rounded-full border border-border flex items-center justify-center">
                    <span className="text-xs">i</span>
                  </div>
                </div>
              </div>

              {models.map((model) => {
                const isSelected = model.id === selectedModel;
                
                return (
                  <SelectPrimitive.Item 
                    key={model.id} 
                    value={model.id}
                    className="flex items-start gap-0 p-0 rounded-lg hover:bg-muted cursor-pointer data-[highlighted]:bg-muted border-none focus:bg-muted mb-1 last:mb-0 outline-none"
                  >
                    <SelectPrimitive.ItemText asChild>
                      <div className="flex items-start justify-between w-full p-3 rounded-lg">
                        <div className="flex items-start gap-3 flex-1">
                          <div className="p-2 bg-secondary rounded-lg">{model.icon}</div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-foreground font-medium text-sm">{model.name}</span>
                            </div>
                            <div className="text-muted-foreground text-sm leading-relaxed">{model.description}</div>
                          </div>
                        </div>
                        {isSelected && !model.isDownload && (
                          <div className="flex-shrink-0 ml-3 mt-0.5">
                            <div className="w-2 h-2 bg-foreground rounded-full" />
                          </div>
                        )}
                      </div>
                    </SelectPrimitive.ItemText>
                  </SelectPrimitive.Item>
                );
              })}

              <div className="mt-2 pt-2 border-t border-border">
                <div className="px-3 py-2 text-muted-foreground text-xs">
                  Select a Lite model to use, or a Pro model to download.
                </div>
              </div>
            </SelectPrimitive.Viewport>
          </SelectPrimitive.Content>
        </SelectPrimitive.Portal>
      </SelectPrimitive.Root>
    </div>
  );
};

export default ModelSelector;
