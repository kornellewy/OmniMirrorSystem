import gradio as gr


# Simple chatbot logic
def chatbot(message, history=[]):
    # Add user message to history
    history.append(("User", message))

    # Generate a simple response (you can make this as complex as you'd like)
    if "hello" in message.lower():
        response = "Hello! How can I assist you today?"
    elif "bye" in message.lower():
        response = "Goodbye! Have a great day!"
    else:
        response = "I'm not sure how to respond to that."

    # Add chatbot response to history
    history.append(("Chatbot", response))

    return history, history  # Return history for both output and state


# Gradio interface
with gr.Blocks() as demo:
    chatbot_interface = gr.Chatbot()  # Chatbot component to display conversation
    message_box = gr.Textbox(
        placeholder="Type your message here..."
    )  # Text input for user
    clear_btn = gr.Button("Clear Chat")  # Button to clear chat

    # Action to take when user submits a message
    def user_message(message, history):
        return chatbot(message, history)

    # Event listeners
    message_box.submit(
        user_message,
        inputs=[message_box, chatbot_interface],
        outputs=[chatbot_interface, chatbot_interface],
    )
    clear_btn.click(lambda: None, None, chatbot_interface)

# Launch the Gradio app
demo.launch()
