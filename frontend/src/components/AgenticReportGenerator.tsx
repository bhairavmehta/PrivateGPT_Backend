import React, { useState, useRef } from 'react';
import { Upload, FileText, Download, Loader2, Bot, BarChart3, Globe, Settings2, ChevronDown, ChevronUp, FileCheck, Target, Zap, FileType } from 'lucide-react';
import EnhancedFileUpload from './EnhancedFileUpload';
import TemplateUpload from './TemplateUpload';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/hooks/use-toast';
import { apiService } from '@/services/api';

interface AgenticReportGeneratorProps {
  className?: string;
}

interface ReportResult {
  success: boolean;
  report_path?: string;
  report_id?: string;
  metadata?: any;
  generation_log?: string[];
  charts_generated?: number;
  sections_generated?: number;
  total_words?: number;
  error?: string;
}

interface SystemStatus {
  available: boolean;
  model_used?: string;
  context_length?: string;
  capabilities?: any;
  supported_report_types?: string[];
  supported_formats?: string[];
  error?: string;
}

const AgenticReportGenerator: React.FC<AgenticReportGeneratorProps> = ({ className }) => {
  // State management
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [templateFiles, setTemplateFiles] = useState<File[]>([]);
  const [textContent, setTextContent] = useState('');
  const [analysisRequirements, setAnalysisRequirements] = useState('');
  const [reportType, setReportType] = useState('comprehensive_analysis');
  const [outputFormat, setOutputFormat] = useState('pdf');
  const [customFilename, setCustomFilename] = useState('');
  const [customRequirements, setCustomRequirements] = useState('');
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  
  // Advanced options
  const [enableWebResearch, setEnableWebResearch] = useState(true);
  const [enableCharts, setEnableCharts] = useState(true);
  const [enableAppendices, setEnableAppendices] = useState(true);
  const [includeResearch, setIncludeResearch] = useState(true);
  
  // UI state
  const [isGenerating, setIsGenerating] = useState(false);
  const [reportResult, setReportResult] = useState<ReportResult | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [generationLog, setGenerationLog] = useState<string[]>([]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();
  
  // Load system status on component mount
  React.useEffect(() => {
    loadSystemStatus();
  }, []);

  const loadSystemStatus = async () => {
    try {
      const response = await fetch('/reports/agentic/status');
      const status = await response.json();
      setSystemStatus(status);
    } catch (error) {
      console.error('Failed to load system status:', error);
      setSystemStatus({ available: false, error: 'Failed to load system status' });
    }
  };

  const generateDocumentAnalysisReport = async () => {
    if (selectedFiles.length === 0) {
      toast({
        title: "No Files Selected",
        description: "Please select at least one file for analysis.",
        variant: "destructive"
      });
      return;
    }

    setIsGenerating(true);
    setReportResult(null);
    setGenerationLog([]);

    try {
      const formData = new FormData();
      
      // Add files
      selectedFiles.forEach(file => {
        formData.append('files', file);
      });
      
      // Add template files
      if (templateFiles.length > 0) {
        templateFiles.forEach(file => {
          formData.append('template_files', file);
        });
      }
      
      // Add parameters
      formData.append('analysis_requirements', analysisRequirements || 'Provide comprehensive analysis of the uploaded documents');
      formData.append('include_charts', enableCharts.toString());
      formData.append('include_research', includeResearch.toString());
      formData.append('output_format', outputFormat);
      if (customFilename) {
        formData.append('custom_filename', customFilename);
      }

      const response = await fetch('/reports/generate/document-analysis', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        // Get metadata from headers
        const reportId = response.headers.get('X-Report-ID');
        const sectionsGenerated = response.headers.get('X-Sections-Generated');
        const chartsGenerated = response.headers.get('X-Charts-Generated');
        const totalWords = response.headers.get('X-Total-Words');
        const processingLog = response.headers.get('X-Processing-Log');

        // Download the file
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = customFilename || `agentic_report_${new Date().getTime()}.${outputFormat}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        // Update result state
        setReportResult({
          success: true,
          report_id: reportId || undefined,
          sections_generated: sectionsGenerated ? parseInt(sectionsGenerated) : undefined,
          charts_generated: chartsGenerated ? parseInt(chartsGenerated) : undefined,
          total_words: totalWords ? parseInt(totalWords) : undefined
        });

        if (processingLog) {
          try {
            const log = JSON.parse(processingLog);
            setGenerationLog(log);
          } catch (e) {
            console.warn('Failed to parse processing log');
          }
        }

        toast({
          title: "Report Generated Successfully",
          description: `Comprehensive analysis report with ${sectionsGenerated || 'multiple'} sections and ${chartsGenerated || '0'} charts has been downloaded.`
        });
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Report generation failed');
      }
    } catch (error) {
      console.error('Document analysis report generation failed:', error);
      setReportResult({
        success: false,
        error: error instanceof Error ? error.message : 'An unexpected error occurred'
      });
      toast({
        title: "Report Generation Failed",
        description: error instanceof Error ? error.message : 'An unexpected error occurred',
        variant: "destructive"
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const generateAgenticTextReport = async () => {
    if (!textContent.trim()) {
      toast({
        title: "No Content Provided",
        description: "Please enter text content for analysis.",
        variant: "destructive"
      });
      return;
    }

    setIsGenerating(true);
    setReportResult(null);
    setGenerationLog(["[System] Initializing agentic report generation..."]);

    const requestBody = {
      content_source: textContent,
      report_type: reportType,
      output_format: outputFormat,
      custom_requirements: customRequirements,
      enable_web_research: enableWebResearch,
      enable_charts: enableCharts,
      enable_appendices: enableAppendices,
      template_name: selectedTemplate || undefined,
    };

    try {
      const response = await fetch('/reports/generate/agentic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        const disposition = response.headers.get('content-disposition');
        let filename = `agentic_report_${new Date().getTime()}.${outputFormat}`;
        if (disposition && disposition.indexOf('attachment') !== -1) {
            const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
            const matches = filenameRegex.exec(disposition);
            if (matches != null && matches[1]) {
                filename = matches[1].replace(/['"]/g, '');
            }
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = customFilename || filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        setReportResult({ success: true });
        toast({
          title: "Agentic Report Generated",
          description: `The report has been downloaded.`
        });
      } else {
        const errorData = await response.json();
        setReportResult({ success: false, error: errorData.error || 'Agentic report generation failed', generation_log: errorData.generation_log });
        if(errorData.generation_log) {
            setGenerationLog(errorData.generation_log);
        }
        throw new Error(errorData.error || 'Agentic report generation failed');
      }
    } catch (error) {
      console.error('Agentic report generation failed:', error);
      toast({
        title: "Generation Failed",
        description: error instanceof Error ? error.message : 'An unexpected error occurred',
        variant: "destructive"
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const reportTypes = [
    { value: 'comprehensive_analysis', label: 'Comprehensive Analysis', description: 'Deep multi-faceted analysis' },
    { value: 'document_analysis', label: 'Document Analysis', description: 'Focused document review' },
    { value: 'research_report', label: 'Research Report', description: 'Academic-style research' },
    { value: 'technical_review', label: 'Technical Review', description: 'Technical assessment' },
    { value: 'data_analysis', label: 'Data Analysis', description: 'Statistical analysis' },
    { value: 'literature_review', label: 'Literature Review', description: 'Literature synthesis' },
    { value: 'market_analysis', label: 'Market Analysis', description: 'Market research' },
    { value: 'compliance_review', label: 'Compliance Review', description: 'Regulatory compliance' }
  ];

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Bot className="w-8 h-8 text-blue-600" />
          <h2 className="text-3xl font-bold tracking-tight">Agentic Report Generator</h2>
          <Zap className="w-8 h-8 text-yellow-500" />
        </div>
        <p className="text-muted-foreground mt-2 max-w-2xl mx-auto">
          Generate comprehensive reports using advanced AI with Qwen2.5-7B-Instruct (100K context), 
          automatic chart generation, web research, and intelligent document analysis.
        </p>
      </div>

      {/* System Status */}
      {systemStatus && (
        <Card className="border-blue-200 bg-blue-50/50">
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="p-2 bg-secondary rounded-md">
                  {/* getIconForStatus(systemStatus.status) is not defined in the original file */}
                </div>
                <span className="font-semibold">{systemStatus.message}</span>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="outline">{systemStatus.time_elapsed.toFixed(2)}s</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="documents" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="documents" className="flex items-center gap-2">
            <Upload className="w-4 h-4" />
            Document Analysis
          </TabsTrigger>
          <TabsTrigger value="text" className="flex items-center gap-2">
            <FileText className="w-4 h-4" />
            Text Analysis
          </TabsTrigger>
        </TabsList>

        {/* Document Analysis Tab */}
        <TabsContent value="documents" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="w-5 h-5" />
                Document Upload & Analysis
              </CardTitle>
              <CardDescription>
                Upload PDF, DOC, DOCX, images, or text files for comprehensive AI-driven analysis with automated report generation.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Enhanced File Upload */}
              <div className="space-y-2">
                <Label>Select Documents & Files</Label>
                <EnhancedFileUpload
                  onFilesSelected={setSelectedFiles}
                  acceptedTypes={['.pdf', '.doc', '.docx', '.txt', '.md', '.jpg', '.jpeg', '.png', '.bmp']}
                  maxFileSize={100}
                  maxFiles={10}
                />
              </div>

              {/* Template Selection */}
              <div className="space-y-2">
                <Label>Template Selection (Optional)</Label>
                <TemplateUpload
                  onTemplateSelect={setSelectedTemplate}
                  selectedTemplate={selectedTemplate}
                />
              </div>

              {/* Analysis Requirements */}
              <div className="space-y-2">
                <Label htmlFor="analysis-requirements">Analysis Requirements</Label>
                <Textarea
                  id="analysis-requirements"
                  placeholder="Describe what kind of analysis you want... (e.g., 'Analyze the financial trends and provide investment recommendations', 'Review the technical documentation for completeness and accuracy')"
                  value={analysisRequirements}
                  onChange={(e) => setAnalysisRequirements(e.target.value)}
                  rows={3}
                />
              </div>

              {/* Advanced Options */}
              <Collapsible open={showAdvancedOptions} onOpenChange={setShowAdvancedOptions}>
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" className="flex items-center gap-2 p-0">
                    <Settings2 className="w-4 h-4" />
                    Advanced Options
                    {showAdvancedOptions ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="space-y-4 mt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="output-format" className="flex items-center gap-2">
                        <FileType className="w-4 h-4" />
                        Output Format
                      </Label>
                      <Select value={outputFormat} onValueChange={setOutputFormat}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="pdf">
                            <div className="flex items-center gap-2">
                              <div className="w-3 h-3 bg-red-500 rounded"></div>
                              <div>
                                <div className="font-medium">PDF</div>
                                <div className="text-xs text-muted-foreground">Professional, print-ready format with charts</div>
                              </div>
                            </div>
                          </SelectItem>
                          <SelectItem value="docx">
                            <div className="flex items-center gap-2">
                              <div className="w-3 h-3 bg-blue-500 rounded"></div>
                              <div>
                                <div className="font-medium">DOCX</div>
                                <div className="text-xs text-muted-foreground">Editable Word document format</div>
                              </div>
                            </div>
                          </SelectItem>
                          <SelectItem value="doc">
                            <div className="flex items-center gap-2">
                              <div className="w-3 h-3 bg-blue-400 rounded"></div>
                              <div>
                                <div className="font-medium">DOC</div>
                                <div className="text-xs text-muted-foreground">Legacy Word format (compatibility)</div>
                              </div>
                            </div>
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="custom-filename">Custom Filename</Label>
                      <Input
                        id="custom-filename"
                        placeholder="my_analysis_report"
                        value={customFilename}
                        onChange={(e) => setCustomFilename(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="enable-charts"
                        checked={enableCharts}
                        onCheckedChange={setEnableCharts}
                      />
                      <Label htmlFor="enable-charts" className="flex items-center gap-2">
                        <BarChart3 className="w-4 h-4" />
                        Generate Charts
                      </Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Switch
                        id="include-research"
                        checked={includeResearch}
                        onCheckedChange={setIncludeResearch}
                      />
                      <Label htmlFor="include-research" className="flex items-center gap-2">
                        <Globe className="w-4 h-4" />
                        Web Research Enhancement
                      </Label>
                    </div>
                  </div>
                </CollapsibleContent>
              </Collapsible>

              {/* Generate Button */}
              <Button 
                onClick={generateDocumentAnalysisReport}
                disabled={selectedFiles.length === 0 || isGenerating || !systemStatus?.available}
                className="w-full"
                size="lg"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Generating Agentic Report...
                  </>
                ) : (
                  <>
                    <Bot className="w-4 h-4 mr-2" />
                    Generate Document Analysis Report
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Text Analysis Tab */}
        <TabsContent value="text" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Text Content Analysis
              </CardTitle>
              <CardDescription>
                Paste or type text content for comprehensive AI analysis and report generation.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Text Content */}
              <div className="space-y-2">
                <Label htmlFor="text-content">Text Content</Label>
                <Textarea
                  id="text-content"
                  placeholder="Paste your text content here for analysis... (minimum 50 characters)"
                  value={textContent}
                  onChange={(e) => setTextContent(e.target.value)}
                  rows={8}
                  className="min-h-[200px]"
                />
                <div className="text-sm text-muted-foreground text-right">
                  {textContent.length} characters
                </div>
              </div>

              {/* Report Type */}
              <div className="space-y-2">
                <Label htmlFor="report-type">Analysis Type</Label>
                <Select value={reportType} onValueChange={setReportType}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {reportTypes.map((type) => (
                      <SelectItem key={type.value} value={type.value}>
                        <div>
                          <div className="font-medium">{type.label}</div>
                          <div className="text-xs text-muted-foreground">{type.description}</div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Custom Requirements */}
              <div className="space-y-2">
                <Label htmlFor="custom-requirements">Custom Requirements (Optional)</Label>
                <Textarea
                  id="custom-requirements"
                  placeholder="Specify any particular focus areas, methodologies, or requirements for the analysis..."
                  value={customRequirements}
                  onChange={(e) => setCustomRequirements(e.target.value)}
                  rows={2}
                />
              </div>

              {/* Generate Button */}
              <Button 
                onClick={generateAgenticTextReport}
                disabled={textContent.length < 50 || isGenerating || !systemStatus?.available}
                className="w-full"
                size="lg"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Generating Agentic Report...
                  </>
                ) : (
                  <>
                    <Bot className="w-4 h-4 mr-2" />
                    Generate Agentic Text Report
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Results */}
      {(reportResult || generationLog.length > 0) && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {reportResult?.success ? (
                <>
                  <FileCheck className="w-5 h-5 text-green-600" />
                  Report Generated Successfully
                </>
              ) : (
                <>
                  <FileText className="w-5 h-5 text-red-600" />
                  Generation Status
                </>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {reportResult && (
              <div className="space-y-4">
                {reportResult.success ? (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {reportResult.report_id && (
                      <div className="text-center p-3 bg-blue-50 rounded-lg">
                        <div className="text-sm text-muted-foreground">Report ID</div>
                        <div className="font-mono text-sm">{reportResult.report_id}</div>
                      </div>
                    )}
                    {reportResult.sections_generated && (
                      <div className="text-center p-3 bg-green-50 rounded-lg">
                        <div className="text-sm text-muted-foreground">Sections</div>
                        <div className="text-2xl font-bold text-green-600">{reportResult.sections_generated}</div>
                      </div>
                    )}
                    {reportResult.charts_generated !== undefined && (
                      <div className="text-center p-3 bg-purple-50 rounded-lg">
                        <div className="text-sm text-muted-foreground">Charts</div>
                        <div className="text-2xl font-bold text-purple-600">{reportResult.charts_generated}</div>
                      </div>
                    )}
                    {reportResult.total_words && (
                      <div className="text-center p-3 bg-orange-50 rounded-lg">
                        <div className="text-sm text-muted-foreground">Words</div>
                        <div className="text-2xl font-bold text-orange-600">{reportResult.total_words.toLocaleString()}</div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                    <div className="text-red-800 font-medium">Generation Failed</div>
                    <div className="text-red-600 text-sm">{reportResult.error}</div>
                  </div>
                )}
              </div>
            )}

            {generationLog.length > 0 && (
              <div className="mt-4">
                <div className="text-sm font-medium mb-2">Generation Log</div>
                <div className="bg-gray-50 rounded-lg p-3 max-h-40 overflow-y-auto">
                  {generationLog.map((entry, index) => (
                    <div key={index} className="text-xs text-gray-600 font-mono">
                      {entry}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default AgenticReportGenerator;
