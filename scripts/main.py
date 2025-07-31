#!/usr/bin/env python3
"""
A1111 Ollama Extension - Main Script

This is the main entry point for the A1111 Ollama extension.
Provides integration with Ollama for chat, prompt helper, and database functionality.
"""

import gradio as gr
import sys
import os
from pathlib import Path

# Add the extension directory to the Python path
extension_path = Path(__file__).parent.parent
sys.path.append(str(extension_path))

try:
    from scripts.database import OllamaDatabase
    from scripts.ollama_api import OllamaAPI
except ImportError as e:
    print(f"Warning: Could not import extension modules: {e}")
    OllamaDatabase = None
    OllamaAPI = None

class OllamaExtension:
    """Main extension class for A1111 Ollama integration."""
    
    def __init__(self):
        self.api = None
        self.database = None
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize the API and database components."""
        try:
            if OllamaAPI:
                self.api = OllamaAPI()
                print("Ollama API initialized successfully")
            
            if OllamaDatabase:
                self.database = OllamaDatabase()
                print("Ollama database initialized successfully")
        except Exception as e:
            print(f"Error initializing components: {e}")
    
    def chat_with_ollama(self, message, model="llama2"):
        """Send a chat message to Ollama."""
        if not self.api:
            return "Error: Ollama API not available"
        
        try:
            response = self.api.chat(message, model)
            
            # Save to database if available
            if self.database:
                self.database.save_conversation(message, response, model)
            
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_prompt_suggestions(self, partial_prompt):
        """Get prompt suggestions based on partial input."""
        if not self.database:
            return ["Database not available"]
        
        try:
            return self.database.get_prompt_suggestions(partial_prompt)
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    def create_ui(self):
        """Create the Gradio interface for the extension."""
        with gr.Blocks(title="A1111 Ollama Extension") as interface:
            gr.Markdown("# A1111 Ollama Extension")
            gr.Markdown("Chat with Ollama models and get prompt suggestions.")
            
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column():
                        model_dropdown = gr.Dropdown(
                            choices=["llama2", "codellama", "mistral"],
                            value="llama2",
                            label="Model"
                        )
                        message_input = gr.Textbox(
                            label="Message",
                            placeholder="Type your message here...",
                            lines=3
                        )
                        send_button = gr.Button("Send", variant="primary")
                    
                    with gr.Column():
                        response_output = gr.Textbox(
                            label="Response",
                            interactive=False,
                            lines=10
                        )
                
                send_button.click(
                    fn=self.chat_with_ollama,
                    inputs=[message_input, model_dropdown],
                    outputs=response_output
                )
            
            with gr.Tab("Prompt Helper"):
                with gr.Row():
                    prompt_input = gr.Textbox(
                        label="Partial Prompt",
                        placeholder="Start typing your prompt...",
                        lines=2
                    )
                    suggestions_output = gr.Textbox(
                        label="Suggestions",
                        interactive=False,
                        lines=8
                    )
                
                prompt_input.change(
                    fn=lambda x: "\n".join(self.get_prompt_suggestions(x)),
                    inputs=prompt_input,
                    outputs=suggestions_output
                )
        
        return interface

# Global extension instance
extension = OllamaExtension()

def on_ui_tabs():
    """Called by A1111 to add custom tabs."""
    return [(extension.create_ui(), "Ollama", "ollama_extension")]

def main():
    """Main function for standalone testing."""
    print("Starting A1111 Ollama Extension...")
    interface = extension.create_ui()
    interface.launch()

if __name__ == "__main__":
    main()
