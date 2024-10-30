from sqlalchemy import create_engine, Column, Integer, String, DateTime, ARRAY, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL

Base = declarative_base()

class Article(Base):
    __tablename__ = 'articles'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(String)
    author = Column(String)
    tags = Column(ARRAY(String))
    created_at = Column(DateTime)

    __table_args__ = (
        Index('idx_title', 'title'),
        Index('idx_author', 'author'),
        Index('idx_tags', 'tags'),
    )

def get_engine():
    return create_engine(DATABASE_URL)

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()
