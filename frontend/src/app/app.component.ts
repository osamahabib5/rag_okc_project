import { Component, OnInit, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { ChatService, ChatResponse } from './services/chat.service';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
  structured_data?: any;
  evidence?: Array<{ table: string; id: number }>;
  isLoading?: boolean;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit, AfterViewChecked {
  @ViewChild('messagesContainer') private messagesContainer!: ElementRef;
  
  title = 'AI Engineering Sandbox';
  messages: Message[] = [];
  userInput = '';
  isProcessing = false;
  serverStatus: 'connected' | 'disconnected' | 'checking' = 'checking';
  private shouldScroll = false;

  // Sample questions for quick access
  sampleQuestions = [
    "How many points did the Warriors score against the Sacramento Kings on October 27, 2023?",
    "Who was the leading scorer in the 2023 Christmas Day game between the Los Angeles Lakers and Boston Celtics?",
    "Which team won the 2024 New Year's Eve game between the Thunder and Timberwolves?",
    "How many points did LeBron James score in the LA Lakers' 140-132 victory over the Rockets on January 16, 2023?",
    "Which player had 40 points on 4/9 in the 2023 NBA Season?"
  ];

  constructor(private chatService: ChatService) {}

  ngOnInit(): void {
    this.checkServerHealth();
    this.addWelcomeMessage();
  }

  ngAfterViewChecked(): void {
    if (this.shouldScroll) {
      this.scrollToBottom();
      this.shouldScroll = false;
    }
  }

  checkServerHealth(): void {
    this.serverStatus = 'checking';
    this.chatService.checkHealth().subscribe({
      next: (response) => {
        if (response.status === 'healthy') {
          this.serverStatus = 'connected';
        } else {
          this.serverStatus = 'disconnected';
        }
      },
      error: () => {
        this.serverStatus = 'disconnected';
      }
    });
  }

  addWelcomeMessage(): void {
    this.messages.push({
      sender: 'bot',
      text: 'Welcome! I\'m your NBA Stats assistant. Ask me anything about NBA games from the 2023-24 and 2024-25 seasons!',
      timestamp: new Date()
    });
    this.shouldScroll = true;
  }

  sendMessage(): void {
    const input = this.userInput.trim();
    if (!input || this.isProcessing) {
      return;
    }

    // Add user message
    this.messages.push({
      sender: 'user',
      text: input,
      timestamp: new Date()
    });
    this.userInput = '';
    this.isProcessing = true;
    this.shouldScroll = true;

    // Add loading message
    const loadingMessage: Message = {
      sender: 'bot',
      text: 'Thinking...',
      timestamp: new Date(),
      isLoading: true
    };
    this.messages.push(loadingMessage);
    this.shouldScroll = true;

    // Send to backend
    this.chatService.sendMessage(input).subscribe({
      next: (response: ChatResponse) => {
        // Remove loading message
        this.messages = this.messages.filter(m => !m.isLoading);

        // Add bot response
        this.messages.push({
          sender: 'bot',
          text: response.answer || 'No answer received.',
          timestamp: new Date(),
          structured_data: response.structured_result,
          evidence: response.evidence
        });
        this.isProcessing = false;
        this.shouldScroll = true;
      },
      error: (error) => {
        // Remove loading message
        this.messages = this.messages.filter(m => !m.isLoading);

        this.messages.push({
          sender: 'bot',
          text: `Error: ${error.error?.detail || 'Failed to contact server. Please make sure the backend is running.'}`,
          timestamp: new Date()
        });
        this.isProcessing = false;
        this.shouldScroll = true;
      }
    });
  }

  useSampleQuestion(question: string): void {
    this.userInput = question;
    this.sendMessage();
  }

  clearChat(): void {
    this.messages = [];
    this.addWelcomeMessage();
  }

  private scrollToBottom(): void {
    try {
      if (this.messagesContainer) {
        this.messagesContainer.nativeElement.scrollTop = 
          this.messagesContainer.nativeElement.scrollHeight;
      }
    } catch (err) {
      console.error('Scroll error:', err);
    }
  }

  formatStructuredData(data: any): string {
    if (!data) return '';
    return JSON.stringify(data, null, 2);
  }

  hasStructuredData(msg: Message): boolean {
    return !!(msg.structured_data && Object.keys(msg.structured_data).length > 0);
  }
}