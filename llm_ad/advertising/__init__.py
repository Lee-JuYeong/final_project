from flask import Flask

def create_app():
    app = Flask(__name__)
    with app.app_context():
        from .routes import ad_routes
        app.register_blueprint(ad_routes)
    return app
