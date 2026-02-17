import gradio as gr
from itertools import count
from Renters_Rights.RAG import renters_rights_assistant
from constants import PLACEHOLDER, TITLE, DESCRIPTION, EXAMPLES

_session_counter = count(1)

def main():
    print("Welcome to the Renters' Rights Act Assistant!")

    def chat_function(message, history, thread_id):
        return renters_rights_assistant(message, thread_id)

    def thread_id_generator():
        return next(_session_counter)

    gr.ChatInterface(
        chat_function,
        additional_inputs=[gr.State(thread_id_generator)],
        additional_inputs_accordion=gr.Accordion(visible=False),
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder=PLACEHOLDER, container=False, scale=7),
        title=TITLE,
        description=DESCRIPTION,
        examples=EXAMPLES,
    ).launch(
        # share=True
    )


if __name__ == "__main__":
    main()
