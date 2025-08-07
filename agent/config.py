from dotenv import load_dotenv
import os

def load_env_config():
    load_dotenv()
    return os.environ
