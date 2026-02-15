import gradio as gr
from Renters_Rights.RAG import renters_rights_assistant

def main():
    print("Welcome to the Renters' Rights Act Assistant!")

    def chat_function(message, history):
        return renters_rights_assistant(message)
    
    gr.ChatInterface(
        chat_function,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder="Ask me any question about the Renters' Rights Act", container=False, scale=7),
        title="Renters' Rights Act Assistant",
        description="Ask me any question about the Renters' Rights Act",
        examples=[ 
            "What are the key changes introduced by the Renters' Rights Act?",
            "How long is the notice period for rent arrears?",
            "What happens to landlords if they don't comply with the Act?"
        ],
    ).launch(
        #share=True
    )

if __name__ == "__main__":
    main()