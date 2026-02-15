import gradio as gr
from Renters_Rights.RAG import renters_rights_assistant
from constants import PLACEHOLDER, TITLE, DESCRIPTION, EXAMPLES

def main():
    print("Welcome to the Renters' Rights Act Assistant!")

    def chat_function(message, history):
        return renters_rights_assistant(message)
    
    gr.ChatInterface(
        chat_function,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder=PLACEHOLDER, container=False, scale=7),
        title=TITLE,
        description=DESCRIPTION,
        examples=EXAMPLES,
    ).launch(
        #share=True
    )

if __name__ == "__main__":
    main()