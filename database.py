from sqlalchemy import create_engine, Column, String, Text, Enum, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://medical_user:medical_password123@medical-postgres:5432/medical_app")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id = Column(String, nullable=False)  # Changed from ForeignId to String to match your usage
    system_prompt = Column(Text, nullable=True)
    
    # Relationship to messages
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(Enum("user", "assistant", "system", name="message_role"), nullable=False)
    content = Column(Text, nullable=False)
    
    # Relationship to conversation
    conversation = relationship("Conversation", back_populates="messages")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
