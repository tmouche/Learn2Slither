
import inspect


def where():
    frame = inspect.currentframe().f_back.f_back
    func_name: str = frame.f_code.co_name

    object_name: str = ""
    try:
        object_name = frame.f_locals['self'].__class__.__qualname__
    except:
        object_name = "<module>"
    return f"{object_name}.{func_name}"

class Logger:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def debug(self, msg: str):
        c_from: str = where() 
        print(f"[DEBUG][{c_from}] -> {msg}")

    def info(self, msg: str):
        c_from: str = where()
        print(f"[INFO][{c_from}] -> {msg}")

    def error(self, msg: str):
        c_from: str = where()
        print(f"[ERROR][{c_from}] -> {msg}")

    
