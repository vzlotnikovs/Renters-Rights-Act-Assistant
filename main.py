import gradio as gr

def main():
    print("Welcome to the Renter's Rights Act chatbot!")
    def greet(name, intensity):
        return "Hello, " + name + "!" * int(intensity)

    demo = gr.Interface(
        fn=greet,
        inputs=["text", "slider"],
        outputs=["text"],
        api_name="predict"
    )

    demo.launch(share=True)

if __name__ == "__main__":
    main()
