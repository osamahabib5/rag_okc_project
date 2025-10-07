import { Component, OnInit, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { ChatService, ChatResponse } from './services/chat.service';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
  structured_data?: any;
  evidence?: Array<{ table: string; id: number }>;
  debug_info?: any;
  isLoading?: boolean;
  error?: boolean;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit, AfterViewChecked {
  @ViewChild('messagesContainer') private messagesContainer!: ElementRef;
  
  title = 'NBA Stats Assistant';
  messages: Message[] = [];
  userInput = '';
  isProcessing = false;
  serverStatus: 'connected' | 'disconnected' | 'checking' = 'checking';
  private shouldScroll = false;
  showDebugInfo = false;

  // Sample questions organized by category
  sampleQuestions = [
    {
      category: 'Game Scores',
      questions: [
        "How many points did the Warriors score against the Sacramento Kings on October 27, 2023?",
        "What was the final score of the game between the Mavericks and Hawks on 1-26-24?",
        "Which team won the 2024 New Year's Eve game between the Thunder and Timberwolves?"
      ]
    },
    {
      category: 'Player Performance',
      questions: [
        "Who was the leading scorer in the 2023 Christmas Day game between the Los Angeles Lakers and Boston Celtics?",
        "How many points did LeBron James score in the LA Lakers' 140-132 victory over the Rockets on January 16, 2023?",
        "Which player had 40 points on 4/9 in the 2023 NBA Season?",
        "How many rebounds did Victor Wembanyama have in his NBA debut?"
      ]
    },
    {
      category: 'Special Dates',
      questions: [
        "What were the results of Christmas Day 2023 NBA games?",
        "Who scored the most points on New Year's Day 2024?"
      ]
    }
  ];

  constructor(private chatService: ChatService) {}

  ngOnInit(): void {
    this.checkServerHealth();
    this.addWelcomeMessage();
    
    // Check health periodically
    setInterval(() => this.checkServerHealth(), 30000);
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
        this.serverStatus = response.status === 'healthy' ? 'connected' : 'disconnected';
        
        // Update welcome message with stats if first check
        if (this.messages.length === 1 && response.stats) {
          const statsMessage = `\n\n📊 Database contains:\n• ${response.stats.games.toLocaleString()} games\n• ${response.stats.player_performances.toLocaleString()} player performances`;
          this.messages[0].text += statsMessage;
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
      text: '🏀 Welcome to the NBA Stats Assistant!\n\nI can help you find information about NBA games from the 2023-24 and 2024-25 seasons.\n\nTry asking about:\n• Game scores and results\n• Player performances\n• Special date games (Christmas, New Year\'s, etc.)\n\nSelect a sample question or type your own!',
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
      text: '🔍 Searching database...',
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
          evidence: response.evidence,
          debug_info: response.debug_info
        });
        this.isProcessing = false;
        this.shouldScroll = true;
      },
      error: (error) => {
        // Remove loading message
        this.messages = this.messages.filter(m => !m.isLoading);

        const errorMessage = error.error?.detail || 'Failed to contact server. Please ensure the backend is running on http://localhost:8000';
        
        this.messages.push({
          sender: 'bot',
          text: `❌ Error: ${errorMessage}`,
          timestamp: new Date(),
          error: true
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

  toggleDebugInfo(): void {
    this.showDebugInfo = !this.showDebugInfo;
  }

  private scrollToBottom(): void {
    try {
      if (this.messagesContainer) {
        const element = this.messagesContainer.nativeElement;
        element.scrollTop = element.scrollHeight;
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

  hasDebugInfo(msg: Message): boolean {
    return !!(msg.debug_info && Object.keys(msg.debug_info).length > 0);
  }

  copyToClipboard(text: string): void {
    navigator.clipboard.writeText(text).then(() => {
      console.log('Copied to clipboard');
    });
  }

  handleKeyPress(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }
}