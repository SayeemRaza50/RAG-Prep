from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from utils import setup_logging
from config import EMBEDDINGS_MODEL_NAME
import os
import faiss
import pickle
import os
from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS
from utils import setup_logging

logger = setup_logging()

def initialize_embeddings_model():
    logger.info('Initializing embeddings model...')
    try:
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        logger.info('Embeddings model initialized.')
        return embeddings_model
    except Exception as e:
        logger.error(f'Error initializing embeddings model: {e}')
        raise

def vectorize_content(embeddings_model, contents):
    logger.info('Vectorizing content...')
    batch_size = 32
    embeddings = []
    total_batches = (len(contents) + batch_size - 1) // batch_size
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(contents))
        batch_contents = contents[start_idx:end_idx]
        logger.info(f'Processing batch {i + 1}/{total_batches}')
        try:
            batch_embeddings = embeddings_model.embed_documents(batch_contents)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f'Error vectorizing batch {i + 1}: {e}')
            continue
    logger.info('Content vectorization completed.')
    return embeddings

def store_embeddings(embeddings_model, processed_data):
    logger.info('Storing embeddings in FAISS vector store...')
    try:
        documents = []
        for item in processed_data:
            doc = Document(
                page_content=item['content'],
                metadata={'id': str(item['id'])}
            )
            documents.append(doc)

        vector_store = FAISS.from_documents(documents, embeddings_model)

        index_dir = 'faiss_index'
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        index_path = os.path.join(index_dir, 'index.faiss')
        faiss.write_index(vector_store.index, index_path)

        docstore_path = os.path.join(index_dir, 'docstore.pkl')
        with open(docstore_path, 'wb') as f:
            pickle.dump(vector_store.docstore, f)

        index_to_docstore_id_path = os.path.join(index_dir, 'index_to_docstore_id.pkl')
        with open(index_to_docstore_id_path, 'wb') as f:
            pickle.dump(vector_store.index_to_docstore_id, f)

        logger.info('Embeddings stored successfully.')
    except Exception as e:
        logger.error(f'Error storing embeddings: {e}')
        raise