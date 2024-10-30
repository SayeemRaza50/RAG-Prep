
import os

DB_USERNAME = 'username'  #
DB_PASSWORD = 'password'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'mydatabase'    

DATABASE_URL = f'postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'secret key')  
EMBEDDINGS_MODEL_NAME = 'all-MiniLM-L6-v2'  

LOG_FILE = 'logs/pipeline.log'
LOG_LEVEL = 'INFO'
