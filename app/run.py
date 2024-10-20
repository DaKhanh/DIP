from flask import Flask
from api.routes import api_blueprint

def create_app():
    app = Flask(__name__)
    
    # 注册蓝图，管理API路由
    app.register_blueprint(api_blueprint)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
