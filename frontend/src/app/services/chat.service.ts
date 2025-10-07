import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { map, catchError } from 'rxjs/operators';

import { BaseService } from './base.service';

export interface ChatMessage {
  question: string;
}

export interface ChatResponse {
  answer: string;
  structured_result?: any;
  evidence?: Array<{ table: string; id: number }>;
  debug_info?: any;
}

@Injectable({
  providedIn: 'root'
})
export class ChatService extends BaseService {
  constructor(protected override http: HttpClient) {
    super(http);
  }

  sendMessage(question: string): Observable<ChatResponse> {
    const endpoint = `${this.baseUrl}/chat`;
    return this.post(endpoint, { question }).pipe(
      map((response: any) => response as ChatResponse),
      catchError((error) => {
        console.error('Error sending message:', error);
        throw error;
      })
    );
  }

  checkHealth(): Observable<any> {
    const endpoint = `${this.baseUrl}/health`;
    return this.get(endpoint);
  }
}