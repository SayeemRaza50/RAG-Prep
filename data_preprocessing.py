import re
from datetime import datetime
from models import Article, get_session
from utils import setup_logging

logger = setup_logging()

def clean_text(text):
    if not text:
        return ''
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[^A-Za-z0-9\s]', '', text) 
    text = ' '.join(text.split())  
    return text

def preprocess_data(data):
    logger.info('Starting data preprocessing...')
    processed_data = []
    for item in data:
        try:
            logger.debug(f'Processing article ID: {item.get("id")}')
            item['content'] = clean_text(item.get('content', ''))
            item['title'] = clean_text(item.get('title', ''))

            metadata = item.get('metadata', {})
            author = metadata.get('author', 'Unknown')
            tags = metadata.get('tags', [])
            if not isinstance(tags, list):
                tags = []

            created_at_str = item.get('created_at', '')
            if created_at_str:
                created_at = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%SZ")
            else:
                created_at = None

            processed_item = {
                'id': item.get('id'),
                'title': item['title'],
                'content': item['content'],
                'author': author,
                'tags': tags,
                'created_at': created_at
            }
            processed_data.append(processed_item)
        except Exception as e:
            logger.error(f'Error processing article ID {item.get("id")}: {e}')
            continue
    logger.info('Data preprocessing completed.')
    return processed_data

def insert_data(processed_data):
    logger.info('Inserting data into the database...')
    session = get_session()
    try:
        session.query(Article).delete()
        session.commit()

        articles = []
        for item in processed_data:
            article = Article(
                id=item['id'],
                title=item['title'],
                content=item['content'],
                author=item['author'],
                tags=item['tags'],
                created_at=item['created_at']
            )
            articles.append(article)

        session.bulk_save_objects(articles)
        session.commit()
        logger.info('Data inserted into the database.')
    except Exception as e:
        logger.error(f'Error inserting data into the database: {e}')
        session.rollback()
        raise
    finally:
        session.close()

