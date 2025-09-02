import { useState, useEffect } from 'react';
import { ChatSession, ChatMessage, ModelType } from '@/types/chat';

export const useChat = () => {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);

  useEffect(() => {
    const savedSessions = localStorage.getItem('chatSessions');
    if (savedSessions) {
      const parsed = JSON.parse(savedSessions);
      const sessionsWithDates = parsed.map((session: any) => ({
        ...session,
        createdAt: new Date(session.createdAt),
        lastUpdated: new Date(session.lastUpdated),
        messages: session.messages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }))
      }));
      setSessions(sessionsWithDates);
      
      // Set the most recent session as current if no current session exists
      if (!currentSession && sessionsWithDates.length > 0) {
        setCurrentSession(sessionsWithDates[0]);
      }
    }
  }, []);

  useEffect(() => {
    if (sessions.length > 0) {
    localStorage.setItem('chatSessions', JSON.stringify(sessions));
    }
  }, [sessions]);

  const createNewSession = (model: ModelType = 'vision') => {
    const newSession: ChatSession = {
      id: `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title: 'New Chat',
      messages: [],
      model,
      createdAt: new Date(),
      lastUpdated: new Date()
    };
    setSessions(prev => [newSession, ...prev]);
    setCurrentSession(newSession);
    return newSession;
  };

  const addMessage = (sessionId: string, message: Omit<ChatMessage, 'id' | 'timestamp'>, existingMessageId?: string) => {
    if (existingMessageId) {
      // Update existing message
      setSessions(prev => prev.map(session => {
        if (session.id === sessionId) {
          const updatedSession = {
            ...session,
            messages: session.messages.map(msg => 
              msg.id === existingMessageId 
                ? { ...msg, ...message, id: existingMessageId, timestamp: msg.timestamp }
                : msg
            ),
            lastUpdated: new Date()
          };
          if (currentSession?.id === sessionId) {
            setCurrentSession(updatedSession);
          }
          return updatedSession;
        }
        return session;
      }));
      return { id: existingMessageId, timestamp: new Date(), ...message } as ChatMessage;
    } else {
      // Create new message with unique ID
    const newMessage: ChatMessage = {
      ...message,
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date()
    };

    setSessions(prev => prev.map(session => {
      if (session.id === sessionId) {
        const updatedSession = {
          ...session,
          messages: [...session.messages, newMessage],
          lastUpdated: new Date(),
          title: session.messages.length === 0 ? message.content.slice(0, 30) + '...' : session.title
        };
          
          // Always update current session if this is the target session
        if (currentSession?.id === sessionId) {
          setCurrentSession(updatedSession);
        }
        return updatedSession;
      }
      return session;
    }));

    return newMessage;
    }
  };

  const updateMessage = (sessionId: string, messageId: string, updates: Partial<ChatMessage>) => {
    setSessions(prev => {
      const newSessions = prev.map(session => {
        if (session.id === sessionId) {
          const updatedSession = {
            ...session,
            messages: session.messages.map(msg => {
              if (msg.id === messageId) {
                return { ...msg, ...updates };
              }
              return msg;
            }),
            lastUpdated: new Date()
          };
          
          if (currentSession?.id === sessionId) {
            setCurrentSession(updatedSession);
          }
          return updatedSession;
        }
        return session;
      });
      
      return newSessions;
    });
  };

  const deleteSession = (sessionId: string) => {
    setSessions(prev => prev.filter(session => session.id !== sessionId));
    if (currentSession?.id === sessionId) {
      setCurrentSession(null);
    }
  };

  return {
    sessions,
    currentSession,
    setCurrentSession,
    createNewSession,
    addMessage,
    updateMessage,
    deleteSession,
    setSessions
  };
};
