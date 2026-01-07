
class Logger:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def debug(self, msg):
        print(f"[DEBUG] -> {msg}")

    def info(self, msg):
        print(f"[INFO] -> {msg}")

    def error(self, msg):
        print(f"[ERROR] -> {msg}")

logger = Logger()