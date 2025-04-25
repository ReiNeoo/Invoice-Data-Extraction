import gradio as gr
import requests
import json

# Configuration
# Update this with your RAG system's endpoint
RAG_API_URL = "http://192.168.88.151:8000/query"


def query_rag_system(message):
    """Send a query to the external RAG system API"""
    try:
        response = requests.get(
            RAG_API_URL,
            params={"query": message},
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "answer": f"Error: Received status code {response.status_code} from RAG API",
                "sources": []
            }
    except requests.exceptions.RequestException as e:
        return {
            "answer": f"Error connecting to RAG system: {str(e)}",
            "sources": []
        }


def process_query(message, history):
    """Process the user's message by querying the RAG system"""
    # Get response from RAG system
    result = query_rag_system(message)

    # Extract answer and sources
    answer = result.get("answer", "No answer received")
    sources = result.get("sources", [])

    # Format sources for display if any exist
    if sources:
        sources_text = "\n\nSources:"
        for i, source in enumerate(sources):
            source_info = source.get("document", f"Source {i+1}")
            sources_text += f"\n- {source_info}"

        response = answer + sources_text
    else:
        response = answer

    return response


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Chat Interface")
    gr.Markdown("Ask questions and get answers from your RAG system")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Ask something...", container=False)
    clear_btn = gr.Button("Clear Chat")

    def user(message, history):
        return "", history + [[message, None]]

    def bot(history):
        query = history[-1][0]
        response = process_query(query, history[:-1])
        history[-1][1] = response
        return history

    def clear_chat():
        return None

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    clear_btn.click(clear_chat, None, chatbot)

    # Optional: Add system status indicator
    with gr.Accordion("System Status", open=False):
        test_btn = gr.Button("Test RAG System Connection")
        status_box = gr.Textbox(label="Status", interactive=False)

        def test_connection():
            try:
                response = requests.get(RAG_API_URL.rsplit(
                    '/', 1)[0] + "/status", timeout=5)
                if response.status_code == 200:
                    return "✅ Connected to RAG system"
                else:
                    return f"⚠️ RAG system returned status code {response.status_code}"
            except requests.exceptions.RequestException:
                return "❌ Cannot connect to RAG system"

        test_btn.click(test_connection, None, status_box)

if __name__ == "__main__":
    print("Starting RAG Chat Interface...")
    print(f"Configured to connect to RAG API at: {RAG_API_URL}")
    demo.launch()
