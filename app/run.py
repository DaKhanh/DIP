from flask import Flask
from flask_cors import CORS
from api.routes import api_blueprint

def create_app():
    app = Flask(__name__)
    
    # 启用CORS，允许所有来源（可以根据需求调整）
    CORS(app)

    # 注册蓝图，管理API路由
    app.register_blueprint(api_blueprint)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
