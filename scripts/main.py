import gradio as gr

def launch_ui():
    with gr.Blocks(title="A1111 Ollama Extension") as demo:
        with gr.Tab("Chat"):
            with gr.Accordion("Ollama Settings", open=False):
                gr.Textbox(label="Remote Ollama URL")
                gr.Button("Connect")
                gr.Dropdown(choices=[], label="Select Model")
                gr.Textbox(label="System Prompt")
                gr.Button("Save System Prompt")
                gr.Button("Load System Prompt")
            gr.Chatbot()
            gr.Textbox(label="Message")
            gr.Button("Send")

        with gr.Tab("Prompt Helper"):
            for label in ["Perspective", "Subject", "Pose", "Mood", "Hair Color", "Eye Color", "Clothing Top", "Clothing Bottom", "Accessories", "Background", "Location", "Style", "Diffusion Model"]:
                with gr.Row():
                    gr.Textbox(label=label)
                    gr.Button("Save")
                    gr.Button("Delete")
                    gr.Button("Randomize")
            gr.Dropdown(choices=["SD 1.5", "SDXL", "PONY", "Illustrious"], label="Diffusion Model")

    demo.launch()

# Placeholder: call main unless imported
if __name__ == "__main__":
    launch_ui()
