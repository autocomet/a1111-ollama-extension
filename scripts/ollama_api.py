#!/usr/bin/env python3
"""
A1111 Ollama Extension - API Module

Handles communication with the Ollama API for model interactions.
"""

import requests
import json
import time
from typing import Dict, List, Optional, Union, Generator
from urllib.parse import urljoin

class OllamaAPI:
    """API client for interacting with Ollama."""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 30):
        """Initialize the Ollama API client."""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'A1111-Ollama-Extension/1.0.0'
        })
    
    def _make_request(self, endpoint: str, method: str = 'GET', 
                     data: Optional[Dict] = None, stream: bool = False) -> Union[Dict, Generator]:
        """Make a request to the Ollama API."""
        url = urljoin(self.base_url + '/', endpoint.lstrip('/'))
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=self.timeout, stream=stream)
            elif method.upper() == 'POST':
                response = self.session.post(
                    url, 
                    json=data, 
                    timeout=self.timeout if not stream else None,
                    stream=stream
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            if stream:
                return self._stream_response(response)
            else:
                return response.json()
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Could not connect to Ollama. Make sure Ollama is running.")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {self.timeout} seconds")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {str(e)}")
    
    def _stream_response(self, response) -> Generator[Dict, None, None]:
        """Process streaming response from Ollama."""
        for line in response.iter_lines():
            if line:
                try:
                    yield json.loads(line.decode('utf-8'))
                except json.JSONDecodeError:
                    continue
    
    def ping(self) -> bool:
        """Check if Ollama server is running."""
        try:
            self._make_request('/api/version')
            return True
        except:
            return False
    
    def get_models(self) -> List[Dict]:
        """Get list of available models."""
        try:
            response = self._make_request('/api/tags')
            return response.get('models', [])
        except Exception as e:
            print(f"Error getting models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> Generator[Dict, None, None]:
        """Pull a model from the Ollama registry."""
        data = {'name': model_name}
        return self._make_request('/api/pull', 'POST', data, stream=True)
    
    def chat(self, message: str, model: str = "llama2", 
            system_prompt: Optional[str] = None, 
            conversation_history: Optional[List[Dict]] = None,
            stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Send a chat message to Ollama."""
        
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user message
        messages.append({
            'role': 'user',
            'content': message
        })
        
        data = {
            'model': model,
            'messages': messages,
            'stream': stream
        }
        
        if stream:
            return self._chat_stream(data)
        else:
            response = self._make_request('/api/chat', 'POST', data)
            return response.get('message', {}).get('content', '')
    
    def _chat_stream(self, data: Dict) -> Generator[str, None, None]:
        """Handle streaming chat response."""
        for chunk in self._make_request('/api/chat', 'POST', data, stream=True):
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
    
    def generate(self, prompt: str, model: str = "llama2", 
                system: Optional[str] = None,
                context: Optional[List[int]] = None,
                stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate text using a prompt."""
        
        data = {
            'model': model,
            'prompt': prompt,
            'stream': stream
        }
        
        if system:
            data['system'] = system
        
        if context:
            data['context'] = context
        
        if stream:
            return self._generate_stream(data)
        else:
            response = self._make_request('/api/generate', 'POST', data)
            return response.get('response', '')
    
    def _generate_stream(self, data: Dict) -> Generator[str, None, None]:
        """Handle streaming generate response."""
        for chunk in self._make_request('/api/generate', 'POST', data, stream=True):
            if 'response' in chunk:
                yield chunk['response']
    
    def embed(self, text: str, model: str = "llama2") -> List[float]:
        """Get embeddings for text."""
        data = {
            'model': model,
            'prompt': text
        }
        
        response = self._make_request('/api/embeddings', 'POST', data)
        return response.get('embedding', [])
    
    def create_model(self, name: str, modelfile: str, 
                    base_model: Optional[str] = None) -> Generator[Dict, None, None]:
        """Create a custom model."""
        data = {
            'name': name,
            'modelfile': modelfile
        }
        
        if base_model:
            data['stream'] = True
        
        return self._make_request('/api/create', 'POST', data, stream=True)
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model."""
        try:
            data = {'name': model_name}
            self._make_request('/api/delete', 'DELETE', data)
            return True
        except Exception as e:
            print(f"Error deleting model {model_name}: {e}")
            return False
    
    def copy_model(self, source: str, destination: str) -> bool:
        """Copy a model."""
        try:
            data = {
                'source': source,
                'destination': destination
            }
            self._make_request('/api/copy', 'POST', data)
            return True
        except Exception as e:
            print(f"Error copying model from {source} to {destination}: {e}")
            return False
    
    def show_model_info(self, model_name: str) -> Dict:
        """Get detailed information about a model."""
        try:
            data = {'name': model_name}
            return self._make_request('/api/show', 'POST', data)
        except Exception as e:
            print(f"Error getting model info for {model_name}: {e}")
            return {}
    
    def get_running_models(self) -> List[Dict]:
        """Get list of currently running models."""
        try:
            response = self._make_request('/api/ps')
            return response.get('models', [])
        except Exception as e:
            print(f"Error getting running models: {e}")
            return []
    
    def check_model_exists(self, model_name: str) -> bool:
        """Check if a model exists locally."""
        models = self.get_models()
        for model in models:
            if model.get('name', '').startswith(model_name):
                return True
        return False
    
    def get_model_suggestions(self, partial_name: str = "") -> List[str]:
        """Get model name suggestions based on partial input."""
        models = self.get_models()
        suggestions = []
        
        for model in models:
            model_name = model.get('name', '')
            if partial_name.lower() in model_name.lower():
                # Extract base model name (before the colon)
                base_name = model_name.split(':')[0]
                if base_name not in suggestions:
                    suggestions.append(base_name)
        
        return sorted(suggestions)
    
    def health_check(self) -> Dict[str, Union[bool, str, List]]:
        """Perform a comprehensive health check."""
        health_info = {
            'server_running': False,
            'version': None,
            'models_available': [],
            'running_models': [],
            'error': None
        }
        
        try:
            # Check if server is running
            health_info['server_running'] = self.ping()
            
            if health_info['server_running']:
                # Get version info
                try:
                    version_info = self._make_request('/api/version')
                    health_info['version'] = version_info.get('version', 'unknown')
                except:
                    pass
                
                # Get available models
                health_info['models_available'] = [m.get('name', '') for m in self.get_models()]
                
                # Get running models
                health_info['running_models'] = [m.get('name', '') for m in self.get_running_models()]
        
        except Exception as e:
            health_info['error'] = str(e)
        
        return health_info
    
    def close(self):
        """Close the session."""
        if self.session:
            self.session.close()
    
    def __del__(self):
        """Ensure session is closed on object destruction."""
        self.close()
