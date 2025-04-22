import os
from .api import serve

HOST = "0.0.0.0"
PORT = "8080"
ROOT = None


if __name__ == "__main__":
    try:
        port = int(os.environ.get("APP_PORT", PORT))
    except Exception as err:
        msg = f"Unable to convert port {port} to an integer"
        raise ValueError(msg) from err
    
    root_path = os.environ.get("ROOT_PATH", ROOT)
    root_path = "" if (root_path is None) else str(root_path)
    host = os.environ.get("APP_HOST", HOST)
    serve(port = port, host = host, root_path = root_path)
