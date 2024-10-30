import streamlit as st
from querrying import load_vector_store, create_retriever, get_similar_articles, generate_summaries
from models import get_session, Article
from utils import setup_logging

logger = setup_logging()

def main():
    st.title("Article Retrieval and Summarization")

    menu = ["Home", "Search Articles", "Summarize Articles"]
    choice = st.sidebar.selectbox("Menu", menu)

    vector_store, embeddings_model = load_vector_store()
    retriever = create_retriever(vector_store)

    session = get_session()
    articles = session.query(Article).all()
    article_dict = {article.id: article for article in articles}

    if choice == "Home":
        st.write("Welcome to the Article Retrieval and Summarization App!")
        st.write("Use the menu on the left to navigate.")
    elif choice == "Search Articles":
        st.header("Search Articles")
        query_text = st.text_input("Enter your search query:")
        if st.button("Search"):
            if query_text:
                results = get_similar_articles(query_text, retriever, article_dict)
                if results:
                    for result in results:
                        st.subheader(result['title'])
                        st.write(f"**Author**: {result['author']}")
                        st.write(f"**Tags**: {', '.join(result['tags'])}")
                        st.write(f"**Created At**: {result['created_at']}")
                        st.write(result['content'])
                        st.write("---")
                else:
                    st.write("No articles found.")
            else:
                st.warning("Please enter a search query.")
    elif choice == "Summarize Articles":
        st.header("Summarize Articles")
        query_text = st.text_input("Enter your search query for summarization:")
        if st.button("Summarize"):
            if query_text:
                results = get_similar_articles(query_text, retriever, article_dict)
                if results:
                    contexts = [result['content'] for result in results]
                    try:
                        summaries = generate_summaries(contexts)
                        for idx, summary in enumerate(summaries):
                            st.subheader(f"Summary {idx + 1}")
                            st.write(summary)
                            st.write("---")
                    except Exception as e:
                        st.error(f"An error occurred during summarization: {e}")
                        logger.error(f"Summarization error: {e}")
                else:
                    st.write("No articles found to summarize.")
            else:
                st.warning("Please enter a search query.")

if __name__ == '__main__':
    main()
