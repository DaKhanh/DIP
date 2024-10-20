from flask import Blueprint, request, jsonify
from app.services.faiss_service import chat_with_llm_chain

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the RAG Chatbot API"}), 200

@api_blueprint.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        response = chat_with_llm_chain(question)

        return jsonify({'answer': response['text']}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
