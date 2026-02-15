from flask import Flask
from flask_socketio import SocketIO
from dotenv import load_dotenv
import os

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

PORT = int(os.getenv("FLASK_PORT", 3000))
SECRET_KEY = os.getenv("SECRET_KEY", "dev")

# ----------------------------
# Initialize Flask + SocketIO
# ----------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

# Load historical CSV / engine if needed
from controller.sliding_window import engine, load_history
with app.app_context():
    try:
        load_history()  # Loads preprocessed CSV into engine
        print("[Server] Engine history loaded ✅")
    except Exception as e:
        print("[Server] Failed to load engine history:", e)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Register HTTP routes
from routes.routes import register_routes
register_routes(app)

# Register WebSocket events
from controller.websocketController import register_socket_events
register_socket_events(socketio)

# Optional: merge raw data / history
from controller.merge_raw_data import merge_raw_data
# merge_raw_data()  # uncomment if you want to load merged data at startup

# ----------------------------
# Run server
# ----------------------------
# if __name__ == "__main__":
#     print(f"[Server] Starting on 0.0.0.0:{PORT}")
#     socketio.run(app, host="0.0.0.0", port=PORT)
