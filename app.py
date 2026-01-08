#!/usr/bin/env python3
"""
Cappy Bot - RAG-powered Slack bot for employee handbook questions
Runs on Google Cloud Run with zero-cost hosting
"""
import os
import logging
from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for models (loaded once at startup)
slack_client = None
embedding_model = None
llm_tokenizer = None
llm_model = None
chroma_client = None
chroma_collection = None

def initialize_models():
    """Initialize all models and clients at startup"""
    global slack_client, embedding_model, llm_tokenizer, llm_model
    global chroma_client, chroma_collection

    logger.info("Initializing Slack client...")
    slack_token = os.environ.get('SLACK_BOT_TOKEN')
    if not slack_token:
        raise ValueError("SLACK_BOT_TOKEN environment variable not set")
    slack_client = WebClient(token=slack_token)

    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    logger.info("Loading LLM tokenizer and model...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    llm_model.eval()  # Set to evaluation mode

    logger.info("Connecting to ChromaDB...")
    persist_directory = "/app/chroma_db"
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    chroma_collection = chroma_client.get_collection("handbook")

    logger.info("All models initialized successfully")

def retrieve_relevant_chunks(query, top_k=3):
    """
    Retrieve most relevant chunks from ChromaDB for a given query
    """
    # Generate query embedding
    query_embedding = embedding_model.encode(query).tolist()

    # Query ChromaDB
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Extract documents
    if results and results['documents']:
        chunks = results['documents'][0]
        return chunks
    return []

def generate_response(question, context_chunks):
    """
    Generate response using Qwen2.5-1.5B-Instruct with retrieved context
    """
    # Build context from chunks
    context = "\n\n".join(context_chunks)

    # Create prompt
    prompt = f"""You are a helpful assistant that answers questions about the employee handbook. Use the provided context to answer the question accurately and concisely.

Context from handbook:
{context}

Question: {question}

Answer the question based on the context above. If the context doesn't contain enough information, say so."""

    # Tokenize and generate
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about the employee handbook."},
        {"role": "user", "content": prompt}
    ]

    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = llm_tokenizer([text], return_tensors="pt").to("cpu")

    with torch.no_grad():
        generated_ids = llm_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=llm_tokenizer.eos_token_id
        )

    # Decode response
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()

def process_question(question, channel, thread_ts=None):
    """
    Process a question through the RAG pipeline and post response to Slack
    """
    try:
        logger.info(f"Processing question: {question}")

        # Step 1: Retrieve relevant chunks
        context_chunks = retrieve_relevant_chunks(question, top_k=3)
        logger.info(f"Retrieved {len(context_chunks)} relevant chunks")

        if not context_chunks:
            response_text = "I couldn't find relevant information in the handbook to answer your question. Please contact HR directly."
        else:
            # Step 2: Generate response
            response_text = generate_response(question, context_chunks)
            logger.info("Generated response")

        # Step 3: Post to Slack
        slack_client.chat_postMessage(
            channel=channel,
            text=response_text,
            thread_ts=thread_ts  # Reply in thread if this is part of a thread
        )
        logger.info("Posted response to Slack")

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        # Post error message to Slack
        try:
            slack_client.chat_postMessage(
                channel=channel,
                text="Sorry, I encountered an error processing your question. Please try again or contact HR.",
                thread_ts=thread_ts
            )
        except:
            pass

@app.route('/slack/events', methods=['POST'])
def slack_events():
    """
    Handle Slack events at /slack/events endpoint
    """
    data = request.json

    # Handle URL verification challenge
    if data.get('type') == 'url_verification':
        logger.info("Handling URL verification challenge")
        return jsonify({'challenge': data.get('challenge')})

    # Handle events
    if data.get('type') == 'event_callback':
        event = data.get('event', {})
        event_type = event.get('type')

        # Handle app_mention events
        if event_type == 'app_mention':
            # Extract information
            text = event.get('text', '')
            channel = event.get('channel')
            thread_ts = event.get('thread_ts') or event.get('ts')  # Reply in thread if in thread

            # Remove bot mention from text to get the actual question
            # Slack mentions look like <@U01234ABC>
            import re
            question = re.sub(r'<@[A-Z0-9]+>', '', text).strip()

            if question:
                # Process in background (don't block the response)
                # For production, consider using a task queue
                import threading
                thread = threading.Thread(
                    target=process_question,
                    args=(question, channel, thread_ts)
                )
                thread.start()
            else:
                # No question asked, send help message
                try:
                    slack_client.chat_postMessage(
                        channel=channel,
                        text="Hi! Ask me anything about the employee handbook. For example: 'How many vacation days do I get?' or 'What's the remote work policy?'",
                        thread_ts=thread_ts
                    )
                except Exception as e:
                    logger.error(f"Error sending help message: {str(e)}")

        # Acknowledge receipt immediately
        return jsonify({'status': 'ok'}), 200

    # Unknown event type
    return jsonify({'status': 'ignored'}), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({'message': 'Cappy Bot is running'}), 200

if __name__ == '__main__':
    # Initialize models at startup
    logger.info("Starting Cappy Bot...")
    initialize_models()

    # Get port from environment (Cloud Run provides this)
    port = int(os.environ.get('PORT', 8080))

    # Run Flask app
    logger.info(f"Starting Flask app on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
