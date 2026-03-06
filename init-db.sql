-- Initialize database for Local RAG System
-- This script runs automatically when PostgreSQL container starts

-- Enable pgvector extension (optional, for future use)
CREATE EXTENSION IF NOT EXISTS vector;

-- File hash tracking table for idempotent ingestion
CREATE TABLE IF NOT EXISTS file_hashes (
    id SERIAL PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    file_hash VARCHAR(32) NOT NULL,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    chunk_count INTEGER DEFAULT 0,
    file_size VARCHAR(20),
    content_type VARCHAR(50),
    last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_file_hashes_path ON file_hashes(file_path);
CREATE INDEX IF NOT EXISTS idx_file_hashes_hash ON file_hashes(file_hash);
CREATE INDEX IF NOT EXISTS idx_file_hashes_processed ON file_hashes(processed_at);

-- Chat history table for n8n Postgres Chat Memory
CREATE TABLE IF NOT EXISTS n8n_chat_histories (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    message JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_session ON n8n_chat_histories(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_created ON n8n_chat_histories(created_at);

-- Ingestion logs for monitoring
CREATE TABLE IF NOT EXISTS ingestion_logs (
    id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    status VARCHAR(20) NOT NULL, -- 'started', 'completed', 'failed'
    chunks_created INTEGER DEFAULT 0,
    duration_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_status ON ingestion_logs(status);
CREATE INDEX IF NOT EXISTS idx_ingestion_created ON ingestion_logs(created_at);

-- Query logs for analytics
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    user_query TEXT NOT NULL,
    routing_decision JSONB,
    search_performed BOOLEAN DEFAULT FALSE,
    results_count INTEGER DEFAULT 0,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_session ON query_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_query_created ON query_logs(created_at);

-- Function to clean old chat history (retain last 30 days)
CREATE OR REPLACE FUNCTION cleanup_old_chat_history()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM n8n_chat_histories
    WHERE created_at < NOW() - INTERVAL '30 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get file processing stats
CREATE OR REPLACE FUNCTION get_ingestion_stats()
RETURNS TABLE (
    total_files BIGINT,
    total_chunks BIGINT,
    last_processed TIMESTAMP WITH TIME ZONE,
    avg_chunks_per_file NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_files,
        COALESCE(SUM(chunk_count), 0)::BIGINT as total_chunks,
        MAX(processed_at) as last_processed,
        ROUND(AVG(chunk_count), 2) as avg_chunks_per_file
    FROM file_hashes;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (if using separate app user)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO n8n;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO n8n;

-- Document images table for storing extracted images from documents
CREATE TABLE IF NOT EXISTS document_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_hash TEXT NOT NULL,
    picture_index INT NOT NULL,
    image_data BYTEA NOT NULL,
    mimetype TEXT NOT NULL,
    width INT,
    height INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_images_file_hash ON document_images(file_hash);
CREATE INDEX IF NOT EXISTS idx_document_images_created ON document_images(created_at);

-- Insert a test record to verify setup
INSERT INTO file_hashes (file_path, file_hash, chunk_count, file_size, content_type)
VALUES ('__setup_test__', 'setup_complete', 0, '0 MB', 'test')
ON CONFLICT (file_path) DO UPDATE SET processed_at = NOW();

-- Log setup completion
DO $$
BEGIN
    RAISE NOTICE 'Local RAG database setup completed successfully';
END $$;
