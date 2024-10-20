from flask import Flask
from flask_cors import CORS
from app.api.routes import api_blueprint

def create_app():
    app = Flask(__name__)
    
    CORS(app)

    app.register_blueprint(api_blueprint)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
