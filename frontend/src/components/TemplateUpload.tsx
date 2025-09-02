import React, { useState, useEffect } from 'react';
import { Upload, FileText, Trash2, Eye, Download, Plus, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';

interface TemplateSection {
  title: string;
  level: number;
  content_type: string;
  content_pattern: string;
  order?: number;
}

interface Template {
  name: string;
  file_name: string;
  sections_count: number;
  created_at: string;
  content_types: string[];
  guidelines: string[];
  title_pattern?: string;
  sections?: TemplateSection[];
  formatting_rules?: Record<string, any>;
  metadata?: Record<string, any>;
}

interface TemplateUploadProps {
  onTemplateSelect?: (templateName: string) => void;
  selectedTemplate?: string;
}

const TemplateUpload: React.FC<TemplateUploadProps> = ({ onTemplateSelect, selectedTemplate }) => {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [templateName, setTemplateName] = useState('');
  const [description, setDescription] = useState('');
  const [viewingTemplate, setViewingTemplate] = useState<Template | null>(null);
  const [activeTab, setActiveTab] = useState('upload');

  useEffect(() => {
    loadTemplates();
  }, []);

  const loadTemplates = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/templates/list');
      const data = await response.json();
      
      if (data.status === 'success') {
        setTemplates(data.templates);
      } else {
        setError('Failed to load templates');
      }
    } catch (err) {
      setError('Failed to connect to server');
      console.error('Error loading templates:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      if (!templateName) {
        // Auto-generate template name from filename
        const name = file.name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_');
        setTemplateName(name);
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || !templateName.trim()) {
      setError('Please select a file and enter a template name');
      return;
    }

    try {
      setUploading(true);
      setError(null);
      setSuccess(null);

      const formData = new FormData();
      formData.append('template_file', selectedFile);
      formData.append('template_name', templateName.trim());
      if (description.trim()) {
        formData.append('description', description.trim());
      }

      const response = await fetch('/templates/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.status === 'success') {
        setSuccess(`Template "${templateName}" uploaded and parsed successfully!`);
        setSelectedFile(null);
        setTemplateName('');
        setDescription('');
        
        // Reset file input
        const fileInput = document.getElementById('template-file') as HTMLInputElement;
        if (fileInput) fileInput.value = '';
        
        // Reload templates
        await loadTemplates();
      } else {
        setError(data.detail || 'Upload failed');
      }
    } catch (err) {
      setError('Upload failed. Please try again.');
      console.error('Upload error:', err);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (templateName: string) => {
    if (!confirm(`Are you sure you want to delete template "${templateName}"?`)) {
      return;
    }

    try {
      const response = await fetch(`/templates/${templateName}`, {
        method: 'DELETE',
      });

      const data = await response.json();

      if (data.status === 'success') {
        setSuccess(`Template "${templateName}" deleted successfully`);
        await loadTemplates();
      } else {
        setError(data.detail || 'Delete failed');
      }
    } catch (err) {
      setError('Delete failed. Please try again.');
      console.error('Delete error:', err);
    }
  };

  const handleViewTemplate = async (templateName: string) => {
    try {
      const response = await fetch(`/templates/${templateName}`);
      const data = await response.json();

      if (data.status === 'success') {
        setViewingTemplate(data.template);
      } else {
        setError(data.detail || 'Failed to load template details');
      }
    } catch (err) {
      setError('Failed to load template details');
      console.error('View template error:', err);
    }
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString();
    } catch {
      return 'Unknown';
    }
  };

  const getContentTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      'paragraph': 'bg-blue-100 text-blue-800',
      'list': 'bg-green-100 text-green-800',
      'table': 'bg-purple-100 text-purple-800',
      'heading': 'bg-orange-100 text-orange-800',
      'mixed': 'bg-gray-100 text-gray-800',
      'chart_reference': 'bg-red-100 text-red-800',
    };
    return colors[type] || 'bg-gray-100 text-gray-800';
  };

  const supportedFormats = ['.pdf', '.docx', '.doc', '.txt', '.md'];

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border">
        <div className="p-6 border-b">
          <div className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            <h2 className="text-xl font-semibold">Template Management</h2>
          </div>
        </div>
        
        <div className="p-6">
          <div className="flex space-x-1 mb-6">
            <button
              onClick={() => setActiveTab('upload')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'upload'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Upload Template
            </button>
            <button
              onClick={() => setActiveTab('manage')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'manage'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Manage Templates
            </button>
          </div>

          {activeTab === 'upload' && (
            <div className="space-y-4">
              <div>
                <label htmlFor="template-file" className="block text-sm font-medium text-gray-700 mb-1">
                  Select Template File
                </label>
                <input
                  id="template-file"
                  type="file"
                  accept={supportedFormats.join(',')}
                  onChange={handleFileSelect}
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
                <p className="text-sm text-gray-500 mt-1">
                  Supported formats: {supportedFormats.join(', ')}
                </p>
              </div>

              <div>
                <label htmlFor="template-name" className="block text-sm font-medium text-gray-700 mb-1">
                  Template Name
                </label>
                <input
                  id="template-name"
                  type="text"
                  value={templateName}
                  onChange={(e) => setTemplateName(e.target.value)}
                  placeholder="Enter template name"
                  className="block w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div>
                <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-1">
                  Description (Optional)
                </label>
                <textarea
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Describe this template..."
                  rows={3}
                  className="block w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <button
                onClick={handleUpload}
                disabled={!selectedFile || !templateName.trim() || uploading}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                {uploading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Uploading & Parsing...
                  </>
                ) : (
                  <>
                    <Plus className="w-4 h-4" />
                    Upload Template
                  </>
                )}
              </button>
            </div>
          )}

          {activeTab === 'manage' && (
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">Available Templates</h3>
                <button
                  onClick={loadTemplates}
                  className="px-3 py-1 text-sm border border-gray-300 rounded-lg hover:bg-gray-50"
                >
                  Refresh
                </button>
              </div>

              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 animate-spin" />
                  <span className="ml-2">Loading templates...</span>
                </div>
              ) : templates.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No templates uploaded yet</p>
                  <p className="text-sm">Upload your first template to get started</p>
                </div>
              ) : (
                <div className="grid gap-4">
                  {templates.map((template) => (
                    <div
                      key={template.file_name}
                      className={`border-2 rounded-lg p-4 ${
                        selectedTemplate === template.file_name ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <h4 className="font-semibold">{template.name}</h4>
                            <span className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded">
                              {template.sections_count} sections
                            </span>
                          </div>
                          
                          <div className="flex flex-wrap gap-1 mb-2">
                            {template.content_types.map((type) => (
                              <span
                                key={type}
                                className={`px-2 py-1 text-xs rounded border ${getContentTypeColor(type)}`}
                              >
                                {type}
                              </span>
                            ))}
                          </div>

                          <div className="text-sm text-gray-600 space-y-1">
                            <p>Created: {formatDate(template.created_at)}</p>
                            {template.guidelines.length > 0 && (
                              <div>
                                <p className="font-medium">Guidelines:</p>
                                <ul className="list-disc list-inside text-xs space-y-1">
                                  {template.guidelines.map((guideline, index) => (
                                    <li key={index}>{guideline}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        </div>

                        <div className="flex flex-col gap-2">
                          {onTemplateSelect && (
                            <button
                              onClick={() => onTemplateSelect(template.file_name)}
                              className={`px-3 py-1 text-sm rounded-lg transition-colors ${
                                selectedTemplate === template.file_name
                                  ? 'bg-blue-600 text-white'
                                  : 'border border-gray-300 hover:bg-gray-50'
                              }`}
                            >
                              {selectedTemplate === template.file_name ? "Selected" : "Select"}
                            </button>
                          )}
                          
                          <button
                            onClick={() => handleViewTemplate(template.file_name)}
                            className="px-3 py-1 text-sm border border-gray-300 rounded-lg hover:bg-gray-50"
                          >
                            <Eye className="w-4 h-4" />
                          </button>

                          <button
                            onClick={() => handleDelete(template.file_name)}
                            className="px-3 py-1 text-sm border border-red-300 text-red-600 rounded-lg hover:bg-red-50"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Template Details Modal */}
      {viewingTemplate && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-hidden">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Template Details: {viewingTemplate.name}</h3>
                <button
                  onClick={() => setViewingTemplate(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ×
                </button>
              </div>
            </div>
            
            <div className="p-6 overflow-y-auto max-h-[60vh]">
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">Structure Overview</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Sections:</span> {viewingTemplate.sections_count}
                    </div>
                    <div>
                      <span className="font-medium">Title Pattern:</span> {viewingTemplate.title_pattern}
                    </div>
                  </div>
                </div>

                <hr />

                <div>
                  <h4 className="font-semibold mb-2">Sections</h4>
                  <div className="space-y-2">
                    {viewingTemplate.sections?.map((section, index) => (
                      <div key={index} className="border rounded p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-medium">{section.title}</span>
                          <span className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded">
                            Level {section.level}
                          </span>
                          <span className={`px-2 py-1 text-xs rounded ${getContentTypeColor(section.content_type)}`}>
                            {section.content_type}
                          </span>
                        </div>
                        {section.content_pattern && (
                          <p className="text-xs text-gray-600">
                            Pattern: {section.content_pattern}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {viewingTemplate.guidelines && viewingTemplate.guidelines.length > 0 && (
                  <>
                    <hr />
                    <div>
                      <h4 className="font-semibold mb-2">Content Guidelines</h4>
                      <ul className="list-disc list-inside text-sm space-y-1">
                        {viewingTemplate.guidelines.map((guideline, index) => (
                          <li key={index}>{guideline}</li>
                        ))}
                      </ul>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4 text-red-600" />
            <p className="text-red-800">{error}</p>
          </div>
        </div>
      )}

      {success && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <p className="text-green-800">{success}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default TemplateUpload;
