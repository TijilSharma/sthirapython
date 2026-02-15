from flask import jsonify

def register_routes(app):

    @app.route("/")
    def health_check():
        return jsonify({
            "status": "ML service running",
            "port": 3001
        })
