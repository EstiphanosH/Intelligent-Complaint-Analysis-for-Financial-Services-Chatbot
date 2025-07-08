from src.rag_logic import RAGSystem
import gradio as gr
import time

# Initialize RAG system
rag = RAGSystem()

def respond(query, history):
    """Generate streaming response with Gradio"""
    result = rag.generate_response(query)
    response = result['answer']
    
    # Format sources
    sources = "\n\n## Sources:\n" + "\n".join([
        f"**Product: {meta['product']}**\n{text[:200]}..."
        for text, meta in zip(
            result['sources']['documents'][0],
            result['sources']['metadatas'][0]
        )
    ])
    
    # Stream tokens
    full_response = response + sources
    for i in range(len(full_response)):
        time.sleep(0.02)
        yield full_response[:i+1]

# Build Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏦 CrediTrust Complaint Analyst")
    gr.Markdown("Ask questions about customer feedback across financial products")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Your Question")
    clear = gr.Button("Clear History")
    
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    ).then(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )