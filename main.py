from email.mime import message
import gradio as gr

def main():
    print("Welcome to the Renter's Rights Act chatbot!")

    def placeholder(input_text, chat_history):
        # This is a placeholder function for the chatbot response
        if len([h for h in chat_history if h['role'] == "assistant"]) % 2 == 0:
            return f"Yes, I do think that: {input_text}"
        else:
            return "I don't think so"

    gr.ChatInterface(
        placeholder,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder="Ask me any question about the Renter's Rights Act", container=False, scale=7),
        title="Renter's Rights Act Assistant",
        description="Ask me any question about the Renter's Rights Act",
        examples=["What are my rights as a tenant?", "How do I file a complaint?", "Can my landlord raise my rent?"],
        cache_examples=True,
    ).launch(share=True)

if __name__ == "__main__":
    main()
