import gradio as gr
import requests
import json
import time

# Optional: install sseclient-py if available
try:
    import sseclient
    has_sseclient = True
except ImportError:
    has_sseclient = False

# Configuration
# Update this with your RAG system's endpoint
RAG_API_URL = "http://192.168.88.151:8000/query"


def query_rag_system(message):
    """Send a query to the external RAG system API using GET and handle streaming response"""
    try:
        # Use GET request with query parameter
        params = {"query": message}
        response = requests.get(
            RAG_API_URL,
            params=params,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )

        if response.status_code == 200:
            # Handle streaming response
            full_response = ""
            sources = []

            # For pure text stream (non-SSE)
            if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                    if chunk:
                        full_response += chunk
            # For SSE stream
            else:
                # If sseclient is available, use it
                if has_sseclient:
                    try:
                        client = sseclient.SSEClient(response)
                        for event in client.events():
                            if event.data:
                                try:
                                    # Try to parse as JSON if it's formatted that way
                                    data = json.loads(event.data)
                                    if 'token' in data:
                                        full_response += data['token']
                                    elif 'sources' in data:
                                        sources = data['sources']
                                except json.JSONDecodeError:
                                    # If not JSON, treat as plain text
                                    full_response += event.data
                    except Exception as e:
                        full_response += f"\nError processing stream: {str(e)}"
                # Fallback approach for handling SSE without sseclient
                else:
                    buffer = ""
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            if line.startswith('data:'):
                                data = line[5:].strip()
                                try:
                                    # Try to parse as JSON
                                    json_data = json.loads(data)
                                    if 'token' in json_data:
                                        full_response += json_data['token']
                                    elif 'sources' in json_data:
                                        sources = json_data['sources']
                                except json.JSONDecodeError:
                                    # If not JSON, treat as plain text
                                    full_response += data

            return {
                "answer": full_response,
                "sources": sources
            }
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

    # Add streaming support
    with gr.Accordion("Settings", open=False):
        use_streaming = gr.Checkbox(label="Enable streaming mode", value=True,
                                    info="Real-time display of tokens as they arrive")

    def user(message, history):
        return "", history + [[message, None]]

    def bot(history, stream_enabled):
        query = history[-1][0]

        # For non-streaming mode, just return the complete response
        if not stream_enabled:
            response = process_query(query, history[:-1])
            history[-1][1] = response
            return history

        # For streaming mode, we'll update the UI incrementally
        # First, set an initial empty response
        history[-1][1] = ""
        yield history

        try:
            # Setup GET request with streaming
            params = {"query": query}
            response = requests.get(
                RAG_API_URL,
                params=params,
                stream=True,
                headers={"Accept": "text/event-stream"}
            )

            if response.status_code == 200:
                full_response = ""
                sources = []

                # Non-SSE text stream
                if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                    for chunk in response.iter_content(chunk_size=128, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            history[-1][1] = full_response
                            yield history
                            time.sleep(0.01)  # Small delay for UI updates

                # SSE stream
                else:
                    if has_sseclient:
                        client = sseclient.SSEClient(response)
                        for event in client.events():
                            if event.data:
                                try:
                                    data = json.loads(event.data)
                                    if 'token' in data:
                                        full_response += data['token']
                                        history[-1][1] = full_response
                                        yield history
                                    elif 'sources' in data:
                                        sources = data['sources']
                                except json.JSONDecodeError:
                                    full_response += event.data
                                    history[-1][1] = full_response
                                    yield history
                    else:
                        # Fallback for SSE without sseclient
                        for line in response.iter_lines(decode_unicode=True):
                            if line:
                                if line.startswith('data:'):
                                    data = line[5:].strip()
                                    try:
                                        json_data = json.loads(data)
                                        if 'token' in json_data:
                                            full_response += json_data['token']
                                            history[-1][1] = full_response
                                            yield history
                                        elif 'sources' in json_data:
                                            sources = json_data['sources']
                                    except json.JSONDecodeError:
                                        full_response += data
                                        history[-1][1] = full_response
                                        yield history

                # Add sources at the end if available
                if sources:
                    sources_text = "\n\nSources:"
                    for i, source in enumerate(sources):
                        source_info = source.get("document", f"Source {i+1}")
                        sources_text += f"\n- {source_info}"

                    full_response += sources_text
                    history[-1][1] = full_response
                    yield history
            else:
                error_msg = f"Error: Received status code {response.status_code} from RAG API"
                history[-1][1] = error_msg
                yield history

        except Exception as e:
            error_msg = f"Error connecting to RAG system: {str(e)}"
            history[-1][1] = error_msg
            yield history

    def clear_chat():
        return None

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, use_streaming], chatbot
    )

    clear_btn.click(clear_chat, None, chatbot)

    # Optional: Add system status indicator
    with gr.Accordion("System Status", open=False):
        test_btn = gr.Button("Test RAG System Connection")
        status_box = gr.Textbox(label="Status", interactive=False)

        def test_connection():
            try:
                # Simple test query
                test_query = "test connection"
                params = {"query": test_query}
                response = requests.get(
                    RAG_API_URL, params=params, timeout=5, stream=False)

                if response.status_code == 200:
                    return f"✅ Connected to RAG system (Status {response.status_code})"
                else:
                    return f"⚠️ RAG system returned status code {response.status_code}"
            except requests.exceptions.RequestException as e:
                return f"❌ Cannot connect to RAG system: {str(e)}"

        test_btn.click(test_connection, None, status_box)

if __name__ == "__main__":
    print("Starting RAG Chat Interface...")
    print(f"Configured to connect to RAG API at: {RAG_API_URL}")
    demo.launch()
