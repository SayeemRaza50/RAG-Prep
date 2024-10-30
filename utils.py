import logging
from config import LOG_FILE, LOG_LEVEL

def setup_logging():
    if not LOG_FILE:
        raise ValueError("LOG_FILE is not set in the configuration.")
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), 'INFO'),
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger
