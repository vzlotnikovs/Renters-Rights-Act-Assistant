import gradio as gr
from itertools import count
from typing import Optional, List, Any
from Renters_Rights.RAG import renters_rights_assistant
from constants import PLACEHOLDER, TITLE, DESCRIPTION, EXAMPLES

_session_counter = count(1)


def main() -> None:
    """
    Launches the Gradio chatbot interface for the Renters' Rights Act Assistant.
    """
    print(
        "Welcome to the Renters' Rights Act Assistant! Click the link below to access the chatbot."
    )

    def thread_id_generator() -> str:
        """
        Generates unique thread IDs for chat sessions.
        """
        return str(next(_session_counter))

    def chat_function(
        message: str, history: List[Any], thread_id: Optional[str]
    ) -> str:
        """
        Handles chat messages and generates responses using the RAG assistant.
        """
        try:
            if thread_id is None:
                thread_id = thread_id_generator()
            return renters_rights_assistant(message, thread_id)
        except Exception as e:
            print(f"Unexpected error while answering your question: {e}.")
            return "Unexpected error while answering your question. Please try again."

    try:
        gr.ChatInterface(
            chat_function,
            additional_inputs=[gr.State(thread_id_generator)],
            additional_inputs_accordion=gr.Accordion(visible=False),
            chatbot=gr.Chatbot(height=300),
            textbox=gr.Textbox(placeholder=PLACEHOLDER, container=False, scale=7),
            title=TITLE,
            description=DESCRIPTION,
            examples=EXAMPLES,
        ).launch()
    except Exception as e:
        print(f"Error launching Gradio interface: {e}")


if __name__ == "__main__":
    main()
