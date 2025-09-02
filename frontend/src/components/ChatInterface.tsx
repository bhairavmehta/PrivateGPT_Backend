import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Download, FileText, AlertCircle, CheckCircle, Clock, BarChart, Zap, Menu, Plus, Bot, MessageSquare } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Alert, AlertDescription } from './ui/alert';
import { Progress } from './ui/progress';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';
import ChatMessage from './ChatMessage';
import ModelSelector from './ModelSelector';
import VoiceRecorder from './VoiceRecorder';
import FloatingChatInput from './FloatingChatInput';
import ChatSidebar from './ChatSidebar';
import ThemeToggle from './ThemeToggle';
import { useChat } from '../hooks/useChat';
import { useToast } from '../hooks/use-toast';
import { ModelType, ReportProgress } from '../types/chat';
import { apiService } from '../services/api';

const ChatInterface: React.FC = () => {
  const {
    sessions,
    currentSession,
    setCurrentSession,
    createNewSession,
    addMessage,
    updateMessage,
    deleteSession,
    setSessions
  } = useChat();

  const [settings, setSettings] = useState({
    model: 'vision' as ModelType,
    deepResearch: false,
    webSearch: false
  });

  const [isLoading, setIsLoading] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentSession?.messages]);

  const handleSendMessage = async (content: string, files?: File[]) => {
    if (!content.trim() && (!files || files.length === 0)) return;
    if (isLoading) return;

    // Ensure we have a session - create one if none exists
    let session = currentSession;
    if (!session) {
      session = createNewSession(settings.model);
    }

    // Use the session ID directly - don't override it
    const sessionId = session.id;
    const messageContent = content.trim();

    const userMessage = addMessage(sessionId, {
      content: messageContent,
      isUser: true,
      attachments: files
    });

    setIsLoading(true);

    const assistantMessage = addMessage(sessionId, {
      content: '',
      isUser: false,
      isCodeResponse: false
    });

    setStreamingMessageId(assistantMessage.id);

    try {
      // Special handling for Report Generation mode
      if (settings.model === 'report_generation') {
        updateMessage(sessionId, assistantMessage.id, { 
          content: '🚀 **Starting Enhanced AI Report Generation...**\n\n🤖 Initializing 6 AI agents for comprehensive analysis...\n\n⏳ This may take 15-25 seconds for premium quality content...' 
        });

        // Check if we have files uploaded - if so, use document analysis instead of text analysis
        const hasFiles = files && files.length > 0;
        
        if (hasFiles) {
          // Use enhanced document analysis with preview
          updateMessage(sessionId, assistantMessage.id, { 
            content: '🚀 **Starting Enhanced Document Analysis...**\n\n📄 Processing uploaded documents with AI agents.\n\n🤖 6 AI agents working together:\n• Planning Agent\n• Research Agent\n• Structure Agent\n• Content Generation Agent\n• Quality Assessment Agent\n• Visualization Agent\n\n⏳ Generating comprehensive analysis...' 
          });

          try {
            const formData = new FormData();
            
            // Add files
            files.forEach(file => {
              formData.append('files', file);
            });
            
            // Add template files if present
            const templateFiles = files?.filter(file => {
              const ext = '.' + file.name.split('.').pop()?.toLowerCase();
              return ['.pdf', '.doc', '.docx', '.txt', '.md'].includes(ext);
            }) || [];
            templateFiles.forEach(file => {
              formData.append('template_files', file);
            });
            
            // Add parameters for enhanced generation
            formData.append('analysis_requirements', messageContent || 'Provide comprehensive analysis of the uploaded documents');
            formData.append('include_charts', 'true');
            formData.append('include_research', settings.webSearch.toString());
            formData.append('output_format', 'pdf');
            formData.append('agentic_mode', 'true'); // Use enhanced agentic mode

            const response = await fetch(`${apiService.baseUrl}/reports/generate/document-analysis`, {
              method: 'POST',
              body: formData,
            });

            if (response.ok) {
              // Get enhanced metadata from headers
              const reportId = response.headers.get('X-Report-ID');
              const sectionsGenerated = response.headers.get('X-Sections-Generated');
              const chartsGenerated = response.headers.get('X-Charts-Generated');
              const totalWords = response.headers.get('X-Total-Words');
              const processingTime = response.headers.get('X-Processing-Time');
              const agentsUsed = response.headers.get('X-Agents-Used');
              const researchConducted = response.headers.get('X-Research-Conducted');

              // Create download links
              const pdfLink = `/reports/download/${reportId}/pdf`;
              const docxLink = `/reports/download/${reportId}/docx`;

              // Update chat with enhanced preview and download options
              updateMessage(sessionId, assistantMessage.id, {
                content: `# 📊 Enhanced Document Analysis Report

**Report Generated Successfully!** ✅

## 📋 Report Summary
- **Report ID**: ${reportId || 'Generated'}
- **Documents Processed**: ${files.length} file(s)
- **Sections Generated**: ${sectionsGenerated || 'Multiple'}
- **Charts Created**: ${chartsGenerated || '0'}
- **Total Words**: ${totalWords ? parseInt(totalWords).toLocaleString() : 'N/A'}
- **Processing Time**: ${processingTime || 'N/A'} seconds
- **AI Agents Used**: ${agentsUsed || '6'}
- **Web Research**: ${researchConducted === 'True' ? '✅ Conducted' : '❌ Disabled'}

## 📖 Executive Summary
This comprehensive document analysis report processes your uploaded files using advanced AI agents. The analysis includes content extraction, thematic analysis, strategic insights, and actionable recommendations based on the document content.

## 🎯 Key Analysis Areas
• **Content Synthesis**: Cross-document analysis and integration
• **Thematic Extraction**: Key themes and patterns identification
• **Strategic Insights**: Business and operational implications
• **Risk Assessment**: Potential challenges and opportunities
• **Data Visualization**: Professional charts and graphs
• **Actionable Recommendations**: Specific next steps and decisions

## 📊 Analysis Highlights
• **Multi-Document Processing**: Comprehensive review of ${files.length} document(s)
• **AI-Enhanced Analysis**: ${agentsUsed || '6'} specialized AI agents working together
• **Professional Formatting**: Charts, graphs, and structured presentation
• **Research Integration**: ${researchConducted === 'True' ? 'Real-time web search enhancement' : 'Internal analysis focus'}

## 💼 Strategic Value
1. **Decision Support**: Data-driven insights for informed decisions
2. **Risk Mitigation**: Proactive identification of potential issues
3. **Opportunity Recognition**: Strategic advantages and possibilities
4. **Process Optimization**: Efficiency improvements and best practices
5. **Stakeholder Communication**: Clear, professional presentation

---

## 📥 Download Your Report

Your comprehensive analysis is ready! Choose your preferred format:

[📄 Download PDF Report](${pdfLink}) - Professional PDF with charts and formatting

[📝 Download DOCX Report](${docxLink}) - Editable Word document for customization

---

**Files Analyzed**: ${files.map(f => f.name).join(', ')}

*Report generated on ${new Date().toLocaleString()} using Enhanced AI Multi-Agent System*`
              });

              toast({
                title: "Enhanced Document Analysis Complete!",
                description: `AI agents generated ${sectionsGenerated || 'comprehensive'} sections with ${chartsGenerated || '0'} charts. Download links provided above.`
              });

            } else {
              const errorData = await response.json();
              throw new Error(errorData.detail || 'Document analysis failed');
            }
          } catch (reportError) {
            updateMessage(sessionId, assistantMessage.id, {
              content: `❌ **Enhanced Document Analysis Failed**\n\n**Error**: ${reportError instanceof Error ? reportError.message : 'Unknown error occurred'}\n\n💡 **Troubleshooting:**\n- Check your internet connection\n- Ensure files are readable (PDF, DOCX, TXT, MD)\n- Try with smaller files\n- Verify the backend server is running\n- Contact support if the issue persists`
            });

            toast({
              title: "Document Analysis Failed",
              description: reportError instanceof Error ? reportError.message : 'Unknown error occurred',
              variant: "destructive"
            });
          }
        } else {
          // Enhanced text analysis with preview
          try {
            updateMessage(sessionId, assistantMessage.id, { 
              content: '🚀 **Enhanced AI Text Analysis Starting...**\n\n🤖 **6 AI Agents Initializing:**\n• 🧠 Planning Agent - Analyzing requirements\n• 🔍 Research Agent - Gathering data\n• 📋 Structure Agent - Determining format\n• ✍️ Content Generation Agent - Creating content\n• 🔍 Quality Assessment Agent - Reviewing quality\n• 📊 Visualization Agent - Creating charts\n\n⏳ Generating comprehensive analysis...' 
            });

            // Use the new enhanced report generation endpoint
            const formData = new FormData();
            formData.append('text_content', messageContent);
            formData.append('title', `Enhanced AI Analysis Report - ${new Date().toLocaleDateString()}`);
            formData.append('enable_web_research', settings.webSearch.toString());
            formData.append('enable_deep_analysis', 'true');
            formData.append('chart_theme', 'professional');

            // Add template files if present
            const templateFiles = files?.filter(file => {
              const ext = '.' + file.name.split('.').pop()?.toLowerCase();
              return ['.pdf', '.doc', '.docx', '.txt', '.md'].includes(ext);
            }) || [];
            templateFiles.forEach(file => {
              formData.append('template_files', file);
            });

            // Start async report generation
            const response = await fetch(`${apiService.baseUrl}/reports/generate/enhanced-async`, {
              method: 'POST',
              body: formData,
            });

            if (response.ok) {
              const result = await response.json();

              if (result.success) {
                const reportId = result.report_id;
                
                // Update message to show polling status
                updateMessage(sessionId, assistantMessage.id, {
                  content: `🚀 **Enhanced Report Generation Started!**\n\n📋 **Report ID**: ${reportId}\n\n⏳ **Status**: Initializing AI agents...\n\n🔄 Checking progress every 3 seconds...`
                });

                // Start robust polling for status with progress bar
                let pollAttempts = 0;
                const maxPollAttempts = 200; // Poll for up to 10 minutes (200 * 3 seconds)
                
                const pollStatus = async () => {
                  try {
                    pollAttempts++;
                    
                    const statusResponse = await fetch(`${apiService.baseUrl}/reports/status/${reportId}`, {
                      method: 'GET',
                      headers: {
                        'Content-Type': 'application/json',
                      },
                    });
                    
                    if (statusResponse.ok) {
                      const statusData = await statusResponse.json();
                      
                      if (statusData.status === 'completed') {
                        // Report is complete - show final result
                        const metadata = statusData.progress.result?.metadata || {};
                        const sections = metadata.sections_count || 0;
                        const charts = metadata.charts_count || 0;
                        const totalWords = metadata.total_words || 0;
                        const processingTime = statusData.progress.result?.processing_time || 0;
                        const agentsUsed = statusData.progress.result?.agents_used || 6;

                        const finalContent = `# 📊 Enhanced AI Analysis Report

**Report Generated Successfully!** ✅

## 📋 Report Summary
- **Report ID**: ${reportId}
- **Sections Generated**: ${sections}
- **Charts Created**: ${charts}
- **Total Words**: ${totalWords.toLocaleString()}
- **Processing Time**: ${processingTime.toFixed(1)} seconds
- **AI Agents Used**: ${agentsUsed}
- **Status**: ✅ Complete

## 📖 Executive Summary
This comprehensive report analyzes your content using advanced AI systems and multiple specialized agents. The analysis covers strategic insights, market dynamics, and actionable recommendations.

## 🎯 Key Features
• **Multi-Agent Analysis**: ${agentsUsed} specialized AI agents working together
• **Comprehensive Coverage**: ${sections} detailed sections with professional structure
• **Visual Elements**: ${charts} professional charts and visualizations
• **Research Integration**: Web research and data analysis included
• **Professional Format**: Both PDF and DOCX formats available

---

## 📥 Download Your Report

Your comprehensive analysis is ready! Choose your preferred format:

[📄 Download PDF Report](${apiService.baseUrl}/reports/download/${reportId}/pdf) - Professional PDF with charts and formatting

[📝 Download DOCX Report](${apiService.baseUrl}/reports/download/${reportId}/docx) - Editable Word document

---

*Report generated on ${new Date().toLocaleString()} using Advanced AI Multi-Agent System*`;

                        updateMessage(sessionId, assistantMessage.id, {
                          content: finalContent
                        });

                        toast({
                          title: "Enhanced Report Complete!",
                          description: `${agentsUsed} AI agents generated ${sections} sections with ${charts} charts in ${processingTime.toFixed(1)}s.`
                        });
                        
                        return; // Stop polling
                      } else if (statusData.status === 'error') {
                        throw new Error(statusData.progress?.error_message || statusData.message || 'Report generation failed');
                      } else {
                        // Update progress with detailed progress bar
                        const progress = statusData.progress || {};
                        const overallProgress = progress.overall_progress || 0;
                        const currentPhase = progress.current_phase || 'Processing...';
                        const phases = progress.phases || {};
                        const completedPhases = Object.values(phases).filter((phase: any) => phase.status === 'completed').length;
                        const totalPhases = Object.keys(phases).length || 9;
                        
                        // Create progress bar visualization
                        const progressBarWidth = Math.round(overallProgress);
                        const progressBar = '█'.repeat(Math.round(progressBarWidth / 5)) + '░'.repeat(20 - Math.round(progressBarWidth / 5));
                        
                        // Get current phase details
                        const phaseList = Object.entries(phases).map(([key, phase]: [string, any]) => {
                          const emoji = phase.status === 'completed' ? '✅' : 
                                       phase.status === 'running' ? '🔄' : '⏳';
                          return `${emoji} ${phase.name || key}`;
                        }).join('\n');

                        const elapsedTime = progress.elapsed_time || 0;
                        const estimatedRemaining = progress.estimated_remaining_seconds || 0;

                        const progressContent = `🚀 **Enhanced Report Generation In Progress**

📋 **Report ID**: \`${reportId}\`

## 📊 Progress Overview
**${overallProgress.toFixed(1)}%** Complete (${completedPhases}/${totalPhases} phases)

\`\`\`
${progressBar} ${progressBarWidth}%
\`\`\`

## 🔄 Current Phase
**${currentPhase}**

## 📋 Phase Status
${phaseList}

---

⏱️ **Elapsed Time**: ${Math.round(elapsedTime)}s  
⏳ **Estimated Remaining**: ${Math.round(estimatedRemaining)}s  
🔄 **Auto-refresh**: Every 3 seconds (Attempt ${pollAttempts})  
🤖 **AI Agents**: ${totalPhases} specialized agents working together

*Started: ${new Date(progress.start_time || Date.now()).toLocaleTimeString()}*`;

                        updateMessage(sessionId, assistantMessage.id, {
                          content: progressContent
                        });
                        
                        // Continue polling if we haven't exceeded max attempts
                        if (pollAttempts < maxPollAttempts) {
                          setTimeout(pollStatus, 3000);
                        } else {
                          throw new Error('Report generation timeout - exceeded maximum polling time');
                        }
                      }
                    } else {
                      // Network error - continue polling with exponential backoff
                      console.warn(`Status check failed (${statusResponse.status}), retrying...`);
                      const backoffDelay = Math.min(3000 + (pollAttempts * 1000), 10000); // Max 10 second delay
                      
                      updateMessage(sessionId, assistantMessage.id, {
                        content: `🚀 **Enhanced Report Generation In Progress**

📋 **Report ID**: \`${reportId}\`

## 🔄 Status Check
⚠️ **Network Issue**: Retrying connection...

🔄 **Auto-refresh**: Every ${Math.round(backoffDelay/1000)} seconds (Attempt ${pollAttempts})  
📡 **Connection**: Reconnecting to server...

*Your report is still being generated in the background*`
                      });
                      
                      if (pollAttempts < maxPollAttempts) {
                        setTimeout(pollStatus, backoffDelay);
                      } else {
                        throw new Error('Unable to connect to server after multiple attempts');
                      }
                    }
                  } catch (error) {
                    console.error('Polling error:', error);
                    
                    // For network errors, continue polling with longer delays
                    if (error instanceof TypeError && error.message.includes('fetch')) {
                      console.warn('Network error during polling, retrying...');
                      const backoffDelay = Math.min(5000 + (pollAttempts * 2000), 15000); // Max 15 second delay
                      
                      updateMessage(sessionId, assistantMessage.id, {
                        content: `🚀 **Enhanced Report Generation In Progress**

📋 **Report ID**: \`${reportId}\`

## 🔄 Connection Status
⚠️ **Network Issue**: Temporary connection problem

🔄 **Auto-retry**: Every ${Math.round(backoffDelay/1000)} seconds (Attempt ${pollAttempts})  
📡 **Status**: Reconnecting to server...  
🤖 **Background**: Your report is still being generated

*Please keep this window open - polling will continue automatically*`
                      });
                      
                      if (pollAttempts < maxPollAttempts) {
                        setTimeout(pollStatus, backoffDelay);
                      } else {
                        updateMessage(sessionId, assistantMessage.id, {
                          content: `⚠️ **Report Generation Status Unknown**

📋 **Report ID**: \`${reportId}\`

## 🔄 Connection Issue
After ${pollAttempts} attempts, we're unable to connect to the server. However, your report may still be generating in the background.

## 💡 What to do:
1. **Refresh the page** and try checking the report status manually
2. **Wait a few minutes** and try again - the server may be temporarily busy
3. **Your report ID is**: \`${reportId}\` - save this for later reference

---

*Report generation may still be in progress on the server*`
                        });
                      }
                    } else {
                      // Actual error - stop polling
                      updateMessage(sessionId, assistantMessage.id, {
                        content: `❌ **Report Generation Failed**

📋 **Report ID**: \`${reportId}\`

**Error**: ${error instanceof Error ? error.message : 'Unknown error'}

💡 **Try Again**: You can start a new report generation.`
                      });
                      
                      toast({
                        title: "Report Generation Failed",
                        description: error instanceof Error ? error.message : 'Unknown error occurred',
                        variant: "destructive"
                      });
                    }
                  }
                };

                // Start polling after a short delay
                setTimeout(pollStatus, 2000);

                toast({
                  title: "Report Generation Started!",
                  description: "Your enhanced report is being generated. Progress will be shown above."
                });
              } else {
                throw new Error(result.error || 'Failed to start report generation');
              }
            } else {
              const errorData = await response.json();
              throw new Error(errorData.detail || 'Failed to start report generation');
            }
          } catch (reportError) {
            updateMessage(sessionId, assistantMessage.id, {
              content: `❌ **Enhanced Report Generation Failed**\n\n**Error**: ${reportError instanceof Error ? reportError.message : 'Unknown error occurred'}\n\n💡 **Troubleshooting:**\n- Check your internet connection\n- Try with shorter content (under 10,000 characters)\n- Ensure the backend server is running\n- Verify AI models are loaded\n- Contact support if the issue persists\n\n🔄 **Alternative**: Try using the regular chat mode for simple questions.`
            });

            toast({
              title: "Enhanced Report Generation Failed",
              description: reportError instanceof Error ? reportError.message : 'Unknown error occurred',
              variant: "destructive"
            });
          }
        }
        
        setIsLoading(false);
        setStreamingMessageId(null);
        return; // Exit early for report generation
      }

      // Regular chat handling (non-report generation)
      const hasImages = files && files.some(file => file.type.startsWith('image/'));
      const useSearch = settings.webSearch || settings.deepResearch;

      // Helper function to fetch sources if source_id is provided
      const fetchSourcesIfNeeded = async (chunk: any) => {
        if (chunk.source_id && chunk.source_count > 0) {
          try {
            console.log(`🔍 Fetching sources for ID: ${chunk.source_id}`);
            const sourcesResponse = await apiService.getSources(chunk.source_id);
            if (sourcesResponse.success) {
              console.log(`✅ Retrieved ${sourcesResponse.count} sources`);
              return sourcesResponse.sources;
            } else {
              console.warn(`⚠️ Failed to fetch sources: ${sourcesResponse.error}`);
              return [];
            }
          } catch (error) {
            console.error('❌ Error fetching sources:', error);
            return [];
          }
        }
        // Fallback to legacy sources field
        return chunk.sources || [];
      };

      // New flow for image-based web search
      if (hasImages && useSearch) {
        const imageFile = files.find(file => file.type.startsWith('image/'));
        if (!imageFile) throw new Error("Image file not found.");

        updateMessage(sessionId, assistantMessage.id, { content: '🖼️ **Analyzing image to generate a search query...**' });

        const queryGenResponse = await apiService.analyzeImage(
          imageFile, 
          'Based on the image, generate a concise and effective web search query. The query should focus on the main subject and context. Return ONLY the search query. Avoid using names.',
          8000,  // Updated to 8000
          false,  // Don't use web search for query generation
          false   // Don't use deep research for query generation
        );
        const searchQuery = queryGenResponse.analysis;

        updateMessage(sessionId, assistantMessage.id, { content: `🤖 Based on your image, I'm now searching for: **"${searchQuery}"**` });
        
        const finalAssistantMessage = addMessage(sessionId, { content: '', isUser: false });
        setStreamingMessageId(finalAssistantMessage.id);

        const streamGenerator = apiService.sendChatMessageStream(searchQuery, sessionId, {
          useWebSearch: settings.webSearch,
          useDeepResearch: settings.deepResearch,
          modelType: 'vision',
          maxTokens: 8000,  // Updated to 8000
        });

        let accumulatedResponse = '';
        for await (const chunk of streamGenerator) {
          if (chunk.accumulated_text) {
            accumulatedResponse = chunk.accumulated_text;
            
            // Check if this is the final chunk and fetch sources if needed
            if (chunk.finished) {
              const sources = await fetchSourcesIfNeeded(chunk);
              updateMessage(sessionId, finalAssistantMessage.id, { 
                content: accumulatedResponse, 
                sources: sources,
                search_query: chunk.search_query,
                source_id: chunk.source_id,
                source_count: chunk.source_count,
                search_type: chunk.search_type
              });
              
              // Cleanup sources after fetching
              if (chunk.source_id) {
                setTimeout(() => apiService.cleanupSources(chunk.source_id), 60000); // Cleanup after 1 minute
              }
            } else {
              updateMessage(sessionId, finalAssistantMessage.id, { 
                content: accumulatedResponse 
              });
            }
          }
          if (chunk.finished) {
            break;
          }
        }
      } else if (hasImages || settings.model === 'vision') {
        // Standard vision analysis (no search)
        const imageFiles = files?.filter(file => file.type.startsWith('image/')) || [];
        if (imageFiles.length > 0) {
          const imageFile = imageFiles[0];
          const prompt = messageContent || 'Analyze this image in detail';
          
          const searchMethodText = settings.deepResearch ? 'Deep Research' : (settings.webSearch ? 'Web Search' : 'Standard Analysis');
          updateMessage(sessionId, assistantMessage.id, { content: `🔍 **Analyzing image with Qwen2.5-VL (${searchMethodText})...**` });

          const streamGenerator = apiService.analyzeImageStream(
            imageFile, 
            prompt, 
            8000,  // Updated to 8000
            settings.webSearch,
            settings.deepResearch
          );
          let accumulatedResponse = '';
          for await (const chunk of streamGenerator) {
            if (chunk.accumulated_text) {
              accumulatedResponse = chunk.accumulated_text;
              
              // Check if this is the final chunk and fetch sources if needed
              if (chunk.finished) {
                const sources = await fetchSourcesIfNeeded(chunk);
                const finalContent = `🎯 **Vision Analysis Results:**\n\n${accumulatedResponse}`;
                updateMessage(sessionId, assistantMessage.id, {
                  content: finalContent,
                  sources: sources,
                  search_query: chunk.search_query,
                  source_id: chunk.source_id,
                  source_count: chunk.source_count,
                  search_type: chunk.search_type
                });
                
                // Cleanup sources after fetching
                if (chunk.source_id) {
                  setTimeout(() => apiService.cleanupSources(chunk.source_id), 60000); // Cleanup after 1 minute
                }
              } else {
                updateMessage(sessionId, assistantMessage.id, {
                  content: `🎯 **Vision Analysis Results:**\n\n${accumulatedResponse}`
                });
              }
            }
            if (chunk.finished) {
              break;
            }
          }
        } else {
          // Vision model for text-only chat
          const streamGenerator = apiService.sendChatMessageStream(messageContent, sessionId, {
            modelType: settings.model,
            maxTokens: 8000,  // Updated to 8000
            useWebSearch: settings.webSearch,
            useDeepResearch: settings.deepResearch,
            files: files
          });
          let accumulatedResponse = '';
          for await (const chunk of streamGenerator) {
            if (chunk.accumulated_text) {
              accumulatedResponse = chunk.accumulated_text;
              
              // Check if this is the final chunk and fetch sources if needed
              if (chunk.finished) {
                const sources = await fetchSourcesIfNeeded(chunk);
                const finalContent = `${accumulatedResponse}`; // Removed model name display
                updateMessage(sessionId, assistantMessage.id, { 
                  content: finalContent, 
                  isCodeResponse: settings.model === 'coding', 
                  sources: sources,
                  search_query: chunk.search_query,
                  source_id: chunk.source_id,
                  source_count: chunk.source_count,
                  search_type: chunk.search_type
                });
                
                // Cleanup sources after fetching
                if (chunk.source_id) {
                  setTimeout(() => apiService.cleanupSources(chunk.source_id), 60000); // Cleanup after 1 minute
                }
              } else {
                updateMessage(sessionId, assistantMessage.id, { content: accumulatedResponse });
              }
            }
            if (chunk.finished) {
              break;
            }
          }
        }
      } else {
        // Standard text chat (with or without search)
        const streamGenerator = apiService.sendChatMessageStream(messageContent, sessionId, {
            useWebSearch: settings.webSearch,
            useDeepResearch: settings.deepResearch,
          modelType: settings.model,
        });

        let accumulatedResponse = '';
        for await (const chunk of streamGenerator) {
          if (chunk.accumulated_text) {
            accumulatedResponse = chunk.accumulated_text;
            
            // Check if this is the final chunk and fetch sources if needed
            if (chunk.finished) {
              const sources = await fetchSourcesIfNeeded(chunk);
              const finalContent = `${accumulatedResponse}`; // Removed model name display
              updateMessage(sessionId, assistantMessage.id, { 
                content: finalContent, 
                isCodeResponse: settings.model === 'coding', 
                sources: sources,
                search_query: chunk.search_query,
                source_id: chunk.source_id,
                source_count: chunk.source_count,
                search_type: chunk.search_type
              });
              
              // Cleanup sources after fetching
              if (chunk.source_id) {
                setTimeout(() => apiService.cleanupSources(chunk.source_id), 60000); // Cleanup after 1 minute
              }
            } else {
              updateMessage(sessionId, assistantMessage.id, { 
                content: accumulatedResponse, 
                isCodeResponse: settings.model === 'coding'
              });
            }
          }
          if (chunk.finished) {
            break;
          }
        }
      }
    } catch (error) {
      updateMessage(sessionId, assistantMessage.id, {
        content: `❌ **Error**: ${error instanceof Error ? error.message : 'Unknown error'}`,
        isCodeResponse: false
      });

      toast({
        title: "Error",
        description: error instanceof Error ? error.message : 'Failed to send message',
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
      setStreamingMessageId(null);
    }
  };

  const handleNewChat = () => {
    createNewSession(settings.model);
    setSidebarOpen(false);
  };

  const handleSessionSelect = (session: any) => {
    setCurrentSession(session);
    setSettings(prev => ({ ...prev, model: session.model }));
    setSidebarOpen(false);
  };

  const handleModelChange = (model: ModelType) => {
    setSettings(prev => ({ ...prev, model }));
    if (currentSession) {
      const updatedSession = { ...currentSession, model };
      setCurrentSession(updatedSession);
      setSessions?.(prev => prev.map(s => s.id === currentSession.id ? updatedSession : s));
    }
  };

  return (
    <div className="flex h-screen bg-background dark:bg-background relative overflow-hidden">
      <ChatSidebar
        sessions={sessions}
        currentSession={currentSession}
        onSessionSelect={handleSessionSelect}
        onNewChat={handleNewChat}
        onDeleteSession={deleteSession}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />

      <div className="flex-1 flex flex-col lg:ml-0 relative">
        <div className="bg-background dark:bg-background border-b border-border dark:border-border">
          <div className="flex items-center justify-between p-4">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden p-2 text-muted-foreground hover:text-foreground hover:bg-accent rounded-lg transition-all duration-200"
              >
                <Menu className="w-5 h-5" />
              </button>
              <ModelSelector
                selectedModel={settings.model}
                onModelChange={handleModelChange}
              />
            </div>
            <div className="flex items-center gap-4">
              <ThemeToggle />
            </div>
          </div>
        </div>

        <div className="h-[calc(100vh-5rem)] overflow-hidden flex flex-col relative">
          {currentSession && currentSession.messages.length > 0 ? (
            <>
              <div className="flex-1 overflow-y-auto p-4 pb-40">
                <div className="max-w-4xl mx-auto lg:pr-0">
                  {currentSession.messages.map((message) => (
                    <ChatMessage 
                      key={message.id} 
                      message={message} 
                      isCodeResponse={message.isCodeResponse}
                      isStreaming={streamingMessageId === message.id}
                      sources={message.sources}
                      search_query={message.search_query}
                    />
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center pb-40">
              <div className="text-center max-w-lg">
                <div className="mb-8">
                  <div className="w-24 h-24 mx-auto bg-card rounded-full flex items-center justify-center shadow-sm border border-border">
                    <MessageSquare className="w-12 h-12 text-muted-foreground" />
                  </div>
                </div>
                <h2 className="text-2xl font-semibold text-foreground mb-4">
                  How can I help you today?
                </h2>
                <p className="text-muted-foreground mb-8 leading-relaxed">
                  I'm powered by advanced AI models for various tasks.
                </p>
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <button
                    onClick={handleNewChat}
                    className="px-6 py-3 bg-accent hover:bg-accent/80 text-accent-foreground rounded-lg transition-colors font-medium flex items-center gap-2 justify-center border border-border"
                  >
                    <Plus className="w-4 h-4" />
                    Start New Chat
                  </button>
                </div>
              </div>
            </div>
          )}

          <FloatingChatInput
            onSendMessage={handleSendMessage}
            disabled={isLoading}
            deepResearch={settings.deepResearch}
            onToggleDeepResearch={() => setSettings(prev => ({ 
              ...prev, 
              deepResearch: !prev.deepResearch,
              webSearch: !prev.deepResearch ? false : prev.webSearch
            }))}
            webSearch={settings.webSearch}
            onToggleWebSearch={() => setSettings(prev => ({ 
              ...prev, 
              webSearch: !prev.webSearch,
              deepResearch: !prev.webSearch ? false : prev.deepResearch
            }))}
            currentModel={settings.model}
          />
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
