import gradio as gr
from Renters_Rights.RAG import renters_rights_chatbot

def main():
    print("Welcome to the Renters' Rights Act chatbot!")

    def chat_function(message, history):
        return renters_rights_chatbot(message)
    
    gr.ChatInterface(
        chat_function,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder="Ask me any question about the Renters' Rights Act", container=False, scale=7),
        title="Renters' Rights Act Assistant",
        description="Ask me any question about the Renters' Rights Act",
        examples=["How often can a landlord raise the rent?", "What is the notice period for evicting a tenant assuming the landlord wants to sell the property?", "When does the Act take effect?"],
        cache_examples=False,
    ).launch(
        #share=True
    )

if __name__ == "__main__":
    main()