from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class Source(str, Enum):
    email = "email"
    file = "file"
    chat = "chat"


class DocumentMetadata(BaseModel):
    source: Optional[Source] = None
    source_id: Optional[str] = None
    url: Optional[str] = None
    year: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    abstract: Optional[str] = None
    published_in: Optional[str] = None
    filename: Optional[str] = None
    tag: Optional[str] = None


class DocumentChunkMetadata(DocumentMetadata):
    document_id: Optional[str] = None


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    text: str
    chunksize: str
    metadata: DocumentChunkMetadata
    embedding: Optional[List[float]] = None


class DocumentChunkWithScore(DocumentChunk):
    score: float


class Document(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[DocumentMetadata] = None


class DocumentWithChunks(Document):
    chunks: List[DocumentChunk]


class DocumentMetadataFilter(BaseModel):
    document_id: Optional[str] = None
    source_id: Optional[str] = None
    source: Optional[Source] = None
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[str] = None
    # filename: Optional[str] = None
    # url: Optional[str] = None
    tag: Optional[str] = None
    chunksize: Optional[str] = None
    # start_date: Optional[str] = None  # any date string format
    # end_date: Optional[str] = None  # any date string format


class Query(BaseModel):
    query: str
    filter_in: Optional[DocumentMetadataFilter] = None
    filter_out: Optional[DocumentMetadataFilter] = None
    top_k: Optional[int] = 3


class QueryWithEmbedding(Query):
    embedding: List[float]


class QueryResult(BaseModel):
    query: str
    results: List[DocumentChunkWithScore]
