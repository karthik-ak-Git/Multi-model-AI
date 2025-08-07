# gradio_app.py

import os
import gradio as gr
from tools.upload_and_ingest import ingest_file
from tools.chat_with_uploaded_docs import ask_question_from_uploaded_doc

USER_ID = "user_001"
uploaded_doc_path = ""


def upload_and_ingest(file):
    global uploaded_doc_path
    if file is None:
        return "⚠️ Please upload a file."

    uploaded_doc_path = file.name
    ingest_file(uploaded_doc_path, user_id=USER_ID)
    return f"✅ File '{file.name}' ingested successfully!"


def chat_with_doc(message, history):
    if not uploaded_doc_path:
        return "⚠️ Please upload and ingest a document first."
    if not message.strip():
        return "⚠️ Please enter a valid question."

    response = ask_question_from_uploaded_doc(message, user_id=USER_ID)
    return response


with gr.Blocks(title="🧠 DocChat Agent") as demo:
    gr.Markdown(
        "# 🤖 Chat with Your Document\nUpload a file and ask questions about it.")

    with gr.Row():
        file_input = gr.File(
            label="📄 Upload a TXT or PDF file", file_types=[".txt", ".pdf"]
        )
        upload_btn = gr.Button("📥 Upload & Ingest")
        upload_output = gr.Textbox(label="Status", interactive=False)

    upload_btn.click(fn=upload_and_ingest, inputs=[
                     file_input], outputs=[upload_output])

    gr.ChatInterface(
        fn=chat_with_doc,
        chatbot=gr.Chatbot(label="🧾 DocBot"),
        textbox=gr.Textbox(
            placeholder="Ask something about your document...", label="❓ Your Question"),
        title="📚 Document Chat",
        description="Upload a document above and chat below.",
        theme="soft",
    )

if __name__ == "__main__":
    demo.launch()
