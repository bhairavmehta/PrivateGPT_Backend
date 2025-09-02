
import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Square } from 'lucide-react';

interface VoiceRecorderProps {
  onTranscription: (text: string) => void;
  disabled?: boolean;
}

const VoiceRecorder: React.FC<VoiceRecorderProps> = ({ onTranscription, disabled }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  useEffect(() => {
    // Check if browser supports speech recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join(' ');
        
        console.log('Speech recognition result:', transcript);
        onTranscription(transcript);
        setIsTranscribing(false);
      };

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsRecording(false);
        setIsTranscribing(false);
      };

      recognitionRef.current.onend = () => {
        setIsRecording(false);
        setIsTranscribing(false);
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [onTranscription]);

  const startRecording = async () => {
    if (!recognitionRef.current) {
      alert('Speech recognition is not supported in this browser');
      return;
    }

    try {
      // Request microphone permission
      await navigator.mediaDevices.getUserMedia({ audio: true });
      
      setIsRecording(true);
      recognitionRef.current.start();
      console.log('Started voice recording');
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Microphone access denied');
    }
  };

  const stopRecording = () => {
    if (recognitionRef.current && isRecording) {
      setIsTranscribing(true);
      recognitionRef.current.stop();
      console.log('Stopped voice recording');
    }
  };

  const handleClick = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <button
      onClick={handleClick}
      disabled={disabled || isTranscribing}
      className={`p-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
        isRecording 
          ? 'text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-200' 
          : 'text-blue-500 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-200'
      }`}
      aria-label={isRecording ? 'Stop recording' : 'Start voice recording'}
      title={
        isTranscribing 
          ? 'Converting speech to text...' 
          : isRecording 
            ? 'Stop recording' 
            : 'Start voice recording'
      }
    >
      {isTranscribing ? (
        <div className="w-5 h-5 border-2 border-current border-t-transparent rounded-full animate-spin" />
      ) : isRecording ? (
        <Square className="w-5 h-5" />
      ) : (
        <Mic className="w-5 h-5" />
      )}
    </button>
  );
};

export default VoiceRecorder;
