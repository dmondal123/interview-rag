CREATE DATABASE interview_embedding_store;
\c interview_embedding_store;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create your tables here if needed 