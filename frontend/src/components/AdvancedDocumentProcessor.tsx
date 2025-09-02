import React, { useState } from 'react';
import { Upload, FileText, Download, Loader2, Compare, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useToast } from '@/hooks/use-toast';
import { apiService } from '@/services/api';

interface ProcessingResult {
  success: boolean;
  output_path: string;
  original_format: string;
  output_format: string;
  original_size: number;
  output_size: number;
  processing_time: string;
  content_length: number;
}

interface ComparisonResult {
  original_stats: {
    word_count: number;
    char_count: number;
    paragraph_count: number;
  };
  modified_stats: {
    word_count: number;
    char_count: number;
    paragraph_count: number;
  };
  changes: {
    word_count_delta: number;
    char_count_delta: number;
    similarity_score: number;
  };
}

const AdvancedDocumentProcessor: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [modificationPrompt, setModificationPrompt] = useState('');
  const [outputFormat, setOutputFormat] = useState<string>('');
  const [customFilename, setCustomFilename] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const [processingResult, setProcessingResult] = useState<ProcessingResult | null>(null);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResult | null>(null);
  const [originalFile, setOriginalFile] = useState<File | null>(null);
  const [modifiedFile, setModifiedFile] = useState<File | null>(null);
  const { toast } = useToast();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file type
      const allowedTypes = ['.pdf', '.doc', '.docx'];
      const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
      
      if (!allowedTypes.includes(fileExtension)) {
        toast({
          title: "Invalid File Type",
          description: "Please select a PDF, DOC, or DOCX file.",
          variant: "destructive"
        });
        return;
      }
      
      setSelectedFile(file);
      toast({
        title: "File Selected",
        description: `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`
      });
    }
  };

  const handleProcess = async () => {
    if (!selectedFile || !modificationPrompt.trim()) {
      toast({
        title: "Missing Information",
        description: "Please select a file and provide modification instructions.",
        variant: "destructive"
      });
      return;
    }

    setIsProcessing(true);
    setProcessingResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('modification_prompt', modificationPrompt);
      
      if (outputFormat) {
        formData.append('output_format', outputFormat);
      }
      
      if (customFilename.trim()) {
        formData.append('custom_filename', customFilename);
      }

      const response = await fetch(`${apiService.baseURL}/documents/process`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        // Handle file download
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        
        // Get filename from response headers or create one
        const contentDisposition = response.headers.get('content-disposition');
        let filename = 'processed_document';
        if (contentDisposition) {
          const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(contentDisposition);
          if (matches != null && matches[1]) {
            filename = matches[1].replace(/['"]/g, '');
          }
        }
        
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        toast({
          title: "Document Processed Successfully",
          description: "Your modified document has been downloaded."
        });

        // Mock processing result for display
        setProcessingResult({
          success: true,
          output_path: filename,
          original_format: selectedFile.name.split('.').pop() || '',
          output_format: outputFormat || selectedFile.name.split('.').pop() || '',
          original_size: selectedFile.size,
          output_size: blob.size,
          processing_time: new Date().toISOString(),
          content_length: 0
        });

      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Processing failed');
      }

    } catch (error) {
      console.error('Processing error:', error);
      toast({
        title: "Processing Failed",
        description: error instanceof Error ? error.message : 'An unexpected error occurred',
        variant: "destructive"
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleCompare = async () => {
    if (!originalFile || !modifiedFile) {
      toast({
        title: "Missing Files",
        description: "Please select both original and modified files for comparison.",
        variant: "destructive"
      });
      return;
    }

    setIsComparing(true);
    setComparisonResult(null);

    try {
      const formData = new FormData();
      formData.append('original_file', originalFile);
      formData.append('modified_file', modifiedFile);

      const response = await fetch(`${apiService.baseURL}/documents/compare`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        setComparisonResult(result);
        
        toast({
          title: "Comparison Complete",
          description: "Document comparison results are ready."
        });
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Comparison failed');
      }

    } catch (error) {
      console.error('Comparison error:', error);
      toast({
        title: "Comparison Failed",
        description: error instanceof Error ? error.message : 'An unexpected error occurred',
        variant: "destructive"
      });
    } finally {
      setIsComparing(false);
    }
  };

  const predefinedPrompts = [
    "Improve the writing style and make it more professional",
    "Summarize the document while keeping key points",
    "Translate this document to a more formal tone",
    "Add executive summary and conclusions",
    "Restructure the content with better headings and organization",
    "Enhance with additional examples and explanations",
    "Convert to a presentation format with bullet points",
    "Fix grammar and spelling errors",
    "Make the language more accessible to general audience",
    "Add references and citations format"
  ];

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold tracking-tight">Advanced Document Processor</h2>
        <p className="text-muted-foreground mt-2">
          Upload PDF/DOC files and modify them using AI-powered analysis and the Qwen2.5-Coder model
        </p>
      </div>

      <Tabs defaultValue="process" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="process">Process Document</TabsTrigger>
          <TabsTrigger value="compare">Compare Documents</TabsTrigger>
        </TabsList>

        <TabsContent value="process" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Document Upload & Processing
              </CardTitle>
              <CardDescription>
                Upload a document and specify how you want it modified. Supports PDF, DOC, and DOCX formats.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* File Upload */}
              <div className="space-y-2">
                <Label htmlFor="file-upload">Select Document</Label>
                <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center">
                  <input
                    id="file-upload"
                    type="file"
                    accept=".pdf,.doc,.docx"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  <div className="space-y-2">
                    <Upload className="w-8 h-8 mx-auto text-muted-foreground" />
                    {selectedFile ? (
                      <div>
                        <p className="font-medium">{selectedFile.name}</p>
                        <p className="text-sm text-muted-foreground">
                          {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                    ) : (
                      <div>
                        <p className="text-muted-foreground">Drop your document here or click to browse</p>
                        <p className="text-xs text-muted-foreground">PDF, DOC, DOCX (Max 100MB)</p>
                      </div>
                    )}
                  </div>
                  <Button
                    variant="outline"
                    className="mt-2"
                    onClick={() => document.getElementById('file-upload')?.click()}
                  >
                    Choose File
                  </Button>
                </div>
              </div>

              {/* Modification Prompt */}
              <div className="space-y-2">
                <Label htmlFor="modification-prompt">Modification Instructions</Label>
                <Textarea
                  id="modification-prompt"
                  placeholder="Describe how you want the document to be modified. For example: 'Make the writing more professional and add an executive summary' or 'Translate to simpler language and add bullet points'"
                  value={modificationPrompt}
                  onChange={(e) => setModificationPrompt(e.target.value)}
                  rows={4}
                />
                
                {/* Predefined Prompts */}
                <div className="space-y-2">
                  <Label className="text-sm">Quick Prompts:</Label>
                  <div className="flex flex-wrap gap-2">
                    {predefinedPrompts.slice(0, 5).map((prompt, index) => (
                      <Button
                        key={index}
                        variant="outline"
                        size="sm"
                        onClick={() => setModificationPrompt(prompt)}
                        className="text-xs"
                      >
                        {prompt}
                      </Button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Output Options */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="output-format">Output Format (Optional)</Label>
                  <Select value={outputFormat} onValueChange={setOutputFormat}>
                    <SelectTrigger>
                      <SelectValue placeholder="Same as input" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">Same as input</SelectItem>
                      <SelectItem value="pdf">PDF</SelectItem>
                      <SelectItem value="docx">DOCX</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="custom-filename">Custom Filename (Optional)</Label>
                  <Input
                    id="custom-filename"
                    placeholder="my_modified_document"
                    value={customFilename}
                    onChange={(e) => setCustomFilename(e.target.value)}
                  />
                </div>
              </div>

              {/* Process Button */}
              <Button 
                onClick={handleProcess} 
                disabled={!selectedFile || !modificationPrompt.trim() || isProcessing}
                className="w-full"
                size="lg"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Processing Document...
                  </>
                ) : (
                  <>
                    <Settings className="w-4 h-4 mr-2" />
                    Process Document
                  </>
                )}
              </Button>

              {/* Processing Result */}
              {processingResult && (
                <Card className="bg-green-50 border-green-200">
                  <CardHeader>
                    <CardTitle className="text-green-800 text-lg">Processing Complete</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    <div className="grid grid-cols-2 gap-2">
                      <span className="text-muted-foreground">Original Format:</span>
                      <span className="font-medium">{processingResult.original_format.toUpperCase()}</span>
                      
                      <span className="text-muted-foreground">Output Format:</span>
                      <span className="font-medium">{processingResult.output_format.toUpperCase()}</span>
                      
                      <span className="text-muted-foreground">Original Size:</span>
                      <span className="font-medium">{(processingResult.original_size / 1024 / 1024).toFixed(2)} MB</span>
                      
                      <span className="text-muted-foreground">Output Size:</span>
                      <span className="font-medium">{(processingResult.output_size / 1024 / 1024).toFixed(2)} MB</span>
                    </div>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="compare" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Compare className="w-5 h-5" />
                Document Comparison
              </CardTitle>
              <CardDescription>
                Compare two documents to see the differences and statistics.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Original File Upload */}
              <div className="space-y-2">
                <Label>Original Document</Label>
                <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-4">
                  <input
                    type="file"
                    accept=".pdf,.doc,.docx"
                    onChange={(e) => setOriginalFile(e.target.files?.[0] || null)}
                    className="w-full"
                  />
                  {originalFile && (
                    <p className="text-sm text-muted-foreground mt-2">
                      {originalFile.name} ({(originalFile.size / 1024 / 1024).toFixed(2)} MB)
                    </p>
                  )}
                </div>
              </div>

              {/* Modified File Upload */}
              <div className="space-y-2">
                <Label>Modified Document</Label>
                <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-4">
                  <input
                    type="file"
                    accept=".pdf,.doc,.docx"
                    onChange={(e) => setModifiedFile(e.target.files?.[0] || null)}
                    className="w-full"
                  />
                  {modifiedFile && (
                    <p className="text-sm text-muted-foreground mt-2">
                      {modifiedFile.name} ({(modifiedFile.size / 1024 / 1024).toFixed(2)} MB)
                    </p>
                  )}
                </div>
              </div>

              {/* Compare Button */}
              <Button 
                onClick={handleCompare} 
                disabled={!originalFile || !modifiedFile || isComparing}
                className="w-full"
                size="lg"
              >
                {isComparing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Comparing Documents...
                  </>
                ) : (
                  <>
                    <Compare className="w-4 h-4 mr-2" />
                    Compare Documents
                  </>
                )}
              </Button>

              {/* Comparison Result */}
              {comparisonResult && (
                <Card className="bg-blue-50 border-blue-200">
                  <CardHeader>
                    <CardTitle className="text-blue-800 text-lg">Comparison Results</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div>
                        <h4 className="font-medium text-sm text-muted-foreground">Original</h4>
                        <div className="space-y-1 text-sm">
                          <p>{comparisonResult.original_stats.word_count} words</p>
                          <p>{comparisonResult.original_stats.paragraph_count} paragraphs</p>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-medium text-sm text-muted-foreground">Modified</h4>
                        <div className="space-y-1 text-sm">
                          <p>{comparisonResult.modified_stats.word_count} words</p>
                          <p>{comparisonResult.modified_stats.paragraph_count} paragraphs</p>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-medium text-sm text-muted-foreground">Changes</h4>
                        <div className="space-y-1 text-sm">
                          <p className={comparisonResult.changes.word_count_delta >= 0 ? "text-green-600" : "text-red-600"}>
                            {comparisonResult.changes.word_count_delta >= 0 ? "+" : ""}{comparisonResult.changes.word_count_delta} words
                          </p>
                          <p>{(comparisonResult.changes.similarity_score * 100).toFixed(1)}% similarity</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AdvancedDocumentProcessor; 