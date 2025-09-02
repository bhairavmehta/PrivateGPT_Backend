import React, { useState, useCallback, useRef } from 'react';
import { Upload, FileText, Image, Trash2, Eye, AlertCircle, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';

interface FileWithPreview extends File {
  preview?: string;
  id: string;
  status: 'uploading' | 'success' | 'error';
  progress: number;
}

interface EnhancedFileUploadProps {
  onFilesSelected: (files: File[]) => void;
  acceptedTypes?: string[];
  maxFileSize?: number; // in MB
  maxFiles?: number;
  className?: string;
}

const EnhancedFileUpload: React.FC<EnhancedFileUploadProps> = ({
  onFilesSelected,
  acceptedTypes = ['.pdf', '.doc', '.docx', '.txt', '.md', '.jpg', '.jpeg', '.png', '.bmp'],
  maxFileSize = 100,
  maxFiles = 10,
  className
}) => {
  const [files, setFiles] = useState<FileWithPreview[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const generateFileId = () => Math.random().toString(36).substr(2, 9);

  const validateFile = (file: File): { valid: boolean; error?: string } => {
    // Check file type
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!acceptedTypes.includes(fileExtension)) {
      return { valid: false, error: `File type ${fileExtension} not supported` };
    }

    // Check file size
    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > maxFileSize) {
      return { valid: false, error: `File size exceeds ${maxFileSize}MB limit` };
    }

    return { valid: true };
  };

  const createFilePreview = async (file: File): Promise<string | undefined> => {
    if (file.type.startsWith('image/')) {
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target?.result as string);
        reader.readAsDataURL(file);
      });
    }
    return undefined;
  };

  const processFiles = async (fileList: File[]) => {
    // Check max files limit
    if (files.length + fileList.length > maxFiles) {
      toast({
        title: "Too many files",
        description: `Maximum ${maxFiles} files allowed`,
        variant: "destructive"
      });
      return;
    }

    const newFiles: FileWithPreview[] = [];

    for (const file of fileList) {
      const validation = validateFile(file);
      
      if (!validation.valid) {
        toast({
          title: "Invalid file",
          description: `${file.name}: ${validation.error}`,
          variant: "destructive"
        });
        continue;
      }

      const preview = await createFilePreview(file);
      const fileWithPreview: FileWithPreview = Object.assign(file, {
        id: generateFileId(),
        preview,
        status: 'success' as const,
        progress: 100
      });

      newFiles.push(fileWithPreview);
    }

    const updatedFiles = [...files, ...newFiles];
    setFiles(updatedFiles);
    onFilesSelected(updatedFiles.map(f => f as File));

    if (newFiles.length > 0) {
      toast({
        title: "Files added",
        description: `${newFiles.length} file(s) added successfully`
      });
    }
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    processFiles(droppedFiles);
  }, [files, onFilesSelected]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    processFiles(selectedFiles);
    
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeFile = (fileId: string) => {
    const updatedFiles = files.filter(f => f.id !== fileId);
    setFiles(updatedFiles);
    onFilesSelected(updatedFiles.map(f => f as File));
    
    toast({
      title: "File removed",
      description: "File removed from upload list"
    });
  };

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('image/')) {
      return <Image className="w-4 h-4" />;
    }
    return <FileText className="w-4 h-4" />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const getFileTypeColor = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf': return 'bg-red-100 text-red-800';
      case 'doc': case 'docx': return 'bg-blue-100 text-blue-800';
      case 'txt': case 'md': return 'bg-gray-100 text-gray-800';
      case 'jpg': case 'jpeg': case 'png': case 'bmp': return 'bg-green-100 text-green-800';
      default: return 'bg-purple-100 text-purple-800';
    }
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive 
            ? 'border-primary bg-primary/5' 
            : 'border-muted-foreground/25 hover:border-muted-foreground/50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={acceptedTypes.join(',')}
          onChange={handleFileInput}
          className="hidden"
        />

        <div className="space-y-4">
          <Upload className={`w-12 h-12 mx-auto ${dragActive ? 'text-primary' : 'text-muted-foreground'}`} />
          
          <div>
            <p className="text-lg font-medium">
              {dragActive ? 'Drop files here' : 'Upload your documents'}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Drag and drop files here, or click to browse
            </p>
            <p className="text-xs text-muted-foreground mt-2">
              Supports: {acceptedTypes.join(', ')} • Max {maxFileSize}MB per file • Up to {maxFiles} files
            </p>
          </div>

          <Button 
            variant="outline" 
            onClick={() => fileInputRef.current?.click()}
            className="mt-4"
          >
            <Upload className="w-4 h-4 mr-2" />
            Choose Files
          </Button>
        </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Selected Files ({files.length}/{maxFiles})
            </CardTitle>
            <CardDescription>
              Review your uploaded files before generating the report
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {files.map((file) => (
              <div key={file.id} className="flex items-center gap-3 p-3 border rounded-lg">
                {/* File Preview */}
                <div className="flex-shrink-0">
                  {file.preview ? (
                    <img 
                      src={file.preview} 
                      alt={file.name}
                      className="w-12 h-12 object-cover rounded border"
                    />
                  ) : (
                    <div className="w-12 h-12 flex items-center justify-center border rounded bg-muted">
                      {getFileIcon(file)}
                    </div>
                  )}
                </div>

                {/* File Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="font-medium truncate">{file.name}</p>
                    <Badge variant="secondary" className={getFileTypeColor(file.name)}>
                      {file.name.split('.').pop()?.toUpperCase()}
                    </Badge>
                  </div>
                  
                  <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
                    <span>{formatFileSize(file.size)}</span>
                    <span>•</span>
                    <span>Modified: {new Date(file.lastModified).toLocaleDateString()}</span>
                  </div>

                  {/* Progress for uploads */}
                  {file.status === 'uploading' && (
                    <Progress value={file.progress} className="mt-2 h-1" />
                  )}
                </div>

                {/* Status & Actions */}
                <div className="flex items-center gap-2">
                  {file.status === 'success' && (
                    <CheckCircle2 className="w-5 h-5 text-green-600" />
                  )}
                  {file.status === 'error' && (
                    <AlertCircle className="w-5 h-5 text-red-600" />
                  )}
                  
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeFile(file.id)}
                    className="text-red-600 hover:text-red-700 hover:bg-red-50"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            ))}

            {/* Summary */}
            <div className="flex justify-between items-center pt-3 border-t text-sm text-muted-foreground">
              <span>
                Total size: {formatFileSize(files.reduce((total, file) => total + file.size, 0))}
              </span>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => {
                  setFiles([]);
                  onFilesSelected([]);
                }}
                className="text-red-600 hover:text-red-700"
              >
                Clear All
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default EnhancedFileUpload; 