import json
from models import Base, get_engine
from utils import setup_logging

logger = setup_logging()

def load_data(file_path):
    logger.info('Loading data from JSON...')
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        logger.info('Data loaded successfully.')
        return data
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        raise

def create_tables():
    logger.info('Creating database tables...')
    try:
        engine = get_engine()
        Base.metadata.create_all(engine)
        logger.info('Database tables created.')
    except Exception as e:
        logger.error(f'Error creating tables: {e}')
        raise
