from data_ingestion import load_data, create_tables
from data_preprocessing import preprocess_data, insert_data
from vectorization import initialize_embeddings_model, vectorize_content, store_embeddings
from utils import setup_logging

def main():
    logger = setup_logging()
    logger.info('Pipeline execution started.')

    create_tables()
    data = load_data('data/data.json')

    processed_data = preprocess_data(data)
    insert_data(processed_data)

    embeddings_model = initialize_embeddings_model()
    contents = [item['content'] for item in processed_data]
    embeddings = vectorize_content(embeddings_model, contents)
    store_embeddings(embeddings_model, processed_data)

    logger.info('Pipeline execution completed.')

if __name__ == '__main__':
    main()
