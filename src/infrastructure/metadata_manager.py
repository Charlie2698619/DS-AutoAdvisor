"""
DS-AutoAdvisor v2.0 Metadata Management System
==============================================

This module provides comprehensive metadata tracking for the DS-AutoAdvisor pipeline.

Features:
- Data lineage tracking across all pipeline stages
- Model performance history storage and retrieval
- Schema registry for data validation
- Execution metadata for pipeline runs
- Performance metrics collection
- Data quality metrics tracking
"""

import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import pandas as pd
import hashlib

class EntityType(Enum):
    DATASET = "dataset"
    MODEL = "model"
    PIPELINE_RUN = "pipeline_run"
    STAGE_EXECUTION = "stage_execution"
    TRANSFORMATION = "transformation"

class LineageType(Enum):
    DERIVED_FROM = "derived_from"
    USED_BY = "used_by"
    GENERATED_BY = "generated_by"
    TRANSFORMED_TO = "transformed_to"

@dataclass
class DatasetMetadata:
    """Metadata for datasets"""
    id: str
    name: str
    path: str
    schema_hash: str
    row_count: int
    column_count: int
    file_size_bytes: int
    created_at: datetime
    updated_at: datetime
    quality_score: float = 0.0
    column_info: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class ModelMetadata:
    """Metadata for trained models"""
    id: str
    name: str
    algorithm: str
    model_path: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_dataset_id: str
    created_at: datetime
    model_size_bytes: int
    training_time_seconds: float
    feature_names: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)

@dataclass
class PipelineRunMetadata:
    """Metadata for pipeline runs"""
    id: str
    pipeline_version: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # "running", "completed", "failed"
    input_dataset_id: str
    configuration: Dict[str, Any]
    stages_completed: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    user_decisions: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class StageExecutionMetadata:
    """Metadata for individual stage executions"""
    id: str
    pipeline_run_id: str
    stage_name: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    input_artifacts: List[str] = field(default_factory=list)
    output_artifacts: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

@dataclass
class LineageRelationship:
    """Relationship between entities in the lineage graph"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: LineageType
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetadataStore:
    """
    Centralized metadata storage and retrieval system
    
    Features:
    - SQLite-based storage with JSON extensions
    - Data lineage tracking
    - Performance metrics collection
    - Schema evolution tracking
    - Query interface for metadata analysis
    """
    
    def __init__(self, db_path: str):
        """
        Initialize metadata store
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enable JSON support
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    schema_hash TEXT NOT NULL,
                    row_count INTEGER,
                    column_count INTEGER,
                    file_size_bytes INTEGER,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    quality_score REAL,
                    column_info TEXT,  -- JSON
                    statistics TEXT    -- JSON
                )
            """)
            
            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    performance_metrics TEXT,  -- JSON
                    hyperparameters TEXT,      -- JSON
                    training_dataset_id TEXT,
                    created_at TIMESTAMP,
                    model_size_bytes INTEGER,
                    training_time_seconds REAL,
                    feature_names TEXT,        -- JSON
                    feature_importance TEXT,   -- JSON
                    FOREIGN KEY (training_dataset_id) REFERENCES datasets (id)
                )
            """)
            
            # Pipeline runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id TEXT PRIMARY KEY,
                    pipeline_version TEXT NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT NOT NULL,
                    input_dataset_id TEXT,
                    configuration TEXT,     -- JSON
                    stages_completed TEXT,  -- JSON
                    error_message TEXT,
                    user_decisions TEXT,    -- JSON
                    FOREIGN KEY (input_dataset_id) REFERENCES datasets (id)
                )
            """)
            
            # Stage executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stage_executions (
                    id TEXT PRIMARY KEY,
                    pipeline_run_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT NOT NULL,
                    input_artifacts TEXT,   -- JSON
                    output_artifacts TEXT,  -- JSON
                    configuration TEXT,     -- JSON
                    metrics TEXT,           -- JSON
                    error_message TEXT,
                    FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs (id)
                )
            """)
            
            # Lineage relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lineage_relationships (
                    id TEXT PRIMARY KEY,
                    source_entity_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    created_at TIMESTAMP,
                    metadata TEXT  -- JSON
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets (created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_created_at ON models (created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_runs_start_time ON pipeline_runs (start_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stage_executions_pipeline_run_id ON stage_executions (pipeline_run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineage_source ON lineage_relationships (source_entity_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineage_target ON lineage_relationships (target_entity_id)")
            
            conn.commit()
            self.logger.info("Metadata store initialized")
    
    def store_dataset_metadata(self, dataset: DatasetMetadata) -> str:
        """Store dataset metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO datasets 
                (id, name, path, schema_hash, row_count, column_count, file_size_bytes,
                 created_at, updated_at, quality_score, column_info, statistics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset.id, dataset.name, dataset.path, dataset.schema_hash,
                dataset.row_count, dataset.column_count, dataset.file_size_bytes,
                dataset.created_at, dataset.updated_at, dataset.quality_score,
                json.dumps(dataset.column_info), json.dumps(dataset.statistics)
            ))
            
            conn.commit()
            self.logger.info(f"Stored dataset metadata: {dataset.id}")
            return dataset.id
    
    def store_model_metadata(self, model: ModelMetadata) -> str:
        """Store model metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO models
                (id, name, algorithm, model_path, performance_metrics, hyperparameters,
                 training_dataset_id, created_at, model_size_bytes, training_time_seconds,
                 feature_names, feature_importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model.id, model.name, model.algorithm, model.model_path,
                json.dumps(model.performance_metrics), json.dumps(model.hyperparameters),
                model.training_dataset_id, model.created_at, model.model_size_bytes,
                model.training_time_seconds, json.dumps(model.feature_names),
                json.dumps(model.feature_importance)
            ))
            
            conn.commit()
            self.logger.info(f"Stored model metadata: {model.id}")
            return model.id
    
    def store_pipeline_run_metadata(self, pipeline_run: PipelineRunMetadata) -> str:
        """Store pipeline run metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO pipeline_runs
                (id, pipeline_version, start_time, end_time, status, input_dataset_id,
                 configuration, stages_completed, error_message, user_decisions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pipeline_run.id, pipeline_run.pipeline_version, pipeline_run.start_time,
                pipeline_run.end_time, pipeline_run.status, pipeline_run.input_dataset_id,
                json.dumps(pipeline_run.configuration), json.dumps(pipeline_run.stages_completed),
                pipeline_run.error_message, json.dumps(pipeline_run.user_decisions)
            ))
            
            conn.commit()
            self.logger.info(f"Stored pipeline run metadata: {pipeline_run.id}")
            return pipeline_run.id
    
    def store_stage_execution_metadata(self, stage_execution: StageExecutionMetadata) -> str:
        """Store stage execution metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO stage_executions
                (id, pipeline_run_id, stage_name, start_time, end_time, status,
                 input_artifacts, output_artifacts, configuration, metrics, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stage_execution.id, stage_execution.pipeline_run_id, stage_execution.stage_name,
                stage_execution.start_time, stage_execution.end_time, stage_execution.status,
                json.dumps(stage_execution.input_artifacts), json.dumps(stage_execution.output_artifacts),
                json.dumps(stage_execution.configuration), json.dumps(stage_execution.metrics),
                stage_execution.error_message
            ))
            
            conn.commit()
            self.logger.info(f"Stored stage execution metadata: {stage_execution.id}")
            return stage_execution.id
    
    def store_lineage_relationship(self, relationship: LineageRelationship) -> str:
        """Store lineage relationship"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO lineage_relationships
                (id, source_entity_id, target_entity_id, relationship_type, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                relationship.id, relationship.source_entity_id, relationship.target_entity_id,
                relationship.relationship_type.value, relationship.created_at,
                json.dumps(relationship.metadata)
            ))
            
            conn.commit()
            self.logger.info(f"Stored lineage relationship: {relationship.id}")
            return relationship.id
    
    def get_dataset_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Retrieve dataset metadata by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()
            
            if row:
                return DatasetMetadata(
                    id=row[0], name=row[1], path=row[2], schema_hash=row[3],
                    row_count=row[4], column_count=row[5], file_size_bytes=row[6],
                    created_at=datetime.fromisoformat(row[7]), updated_at=datetime.fromisoformat(row[8]),
                    quality_score=row[9], column_info=json.loads(row[10] or '{}'),
                    statistics=json.loads(row[11] or '{}')
                )
        return None
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
            row = cursor.fetchone()
            
            if row:
                return ModelMetadata(
                    id=row[0], name=row[1], algorithm=row[2], model_path=row[3],
                    performance_metrics=json.loads(row[4] or '{}'),
                    hyperparameters=json.loads(row[5] or '{}'),
                    training_dataset_id=row[6], created_at=datetime.fromisoformat(row[7]),
                    model_size_bytes=row[8], training_time_seconds=row[9],
                    feature_names=json.loads(row[10] or '[]'),
                    feature_importance=json.loads(row[11] or '{}')
                )
        return None
    
    def get_similar_schemas(self, schema_hash: str, threshold: float = 0.8) -> List[DatasetMetadata]:
        """Get datasets with similar schemas"""
        # For simplicity, using exact match for now
        # In practice, this would use schema similarity algorithms
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM datasets WHERE schema_hash = ?", (schema_hash,))
            rows = cursor.fetchall()
            
            datasets = []
            for row in rows:
                datasets.append(DatasetMetadata(
                    id=row[0], name=row[1], path=row[2], schema_hash=row[3],
                    row_count=row[4], column_count=row[5], file_size_bytes=row[6],
                    created_at=datetime.fromisoformat(row[7]), updated_at=datetime.fromisoformat(row[8]),
                    quality_score=row[9], column_info=json.loads(row[10] or '{}'),
                    statistics=json.loads(row[11] or '{}')
                ))
            
            return datasets
    
    def get_model_performance_history(self, algorithm: str = None, 
                                    dataset_schema_hash: str = None) -> List[ModelMetadata]:
        """Get historical model performance for similar datasets/algorithms"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT m.* FROM models m"
            params = []
            
            if dataset_schema_hash:
                query += " JOIN datasets d ON m.training_dataset_id = d.id"
                
            query += " WHERE 1=1"
            
            if algorithm:
                query += " AND m.algorithm = ?"
                params.append(algorithm)
                
            if dataset_schema_hash:
                query += " AND d.schema_hash = ?"
                params.append(dataset_schema_hash)
                
            query += " ORDER BY m.created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            models = []
            for row in rows:
                models.append(ModelMetadata(
                    id=row[0], name=row[1], algorithm=row[2], model_path=row[3],
                    performance_metrics=json.loads(row[4] or '{}'),
                    hyperparameters=json.loads(row[5] or '{}'),
                    training_dataset_id=row[6], created_at=datetime.fromisoformat(row[7]),
                    model_size_bytes=row[8], training_time_seconds=row[9],
                    feature_names=json.loads(row[10] or '[]'),
                    feature_importance=json.loads(row[11] or '{}')
                ))
            
            return models
    
    def get_data_lineage(self, entity_id: str, depth: int = 3) -> Dict[str, Any]:
        """Get data lineage graph for an entity"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all relationships connected to this entity
            cursor.execute("""
                SELECT * FROM lineage_relationships 
                WHERE source_entity_id = ? OR target_entity_id = ?
            """, (entity_id, entity_id))
            
            relationships = cursor.fetchall()
            
            lineage_graph = {
                "entity_id": entity_id,
                "relationships": [],
                "connected_entities": set()
            }
            
            for rel in relationships:
                relationship = {
                    "id": rel[0],
                    "source_entity_id": rel[1],
                    "target_entity_id": rel[2],
                    "relationship_type": rel[3],
                    "created_at": rel[4],
                    "metadata": json.loads(rel[5] or '{}')
                }
                lineage_graph["relationships"].append(relationship)
                lineage_graph["connected_entities"].add(rel[1])
                lineage_graph["connected_entities"].add(rel[2])
            
            lineage_graph["connected_entities"] = list(lineage_graph["connected_entities"])
            return lineage_graph
    
    def cleanup_old_metadata(self, retention_days: int = 90):
        """Clean up old metadata beyond retention period"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clean up old pipeline runs and their associated stage executions
            cursor.execute("DELETE FROM stage_executions WHERE pipeline_run_id IN (SELECT id FROM pipeline_runs WHERE start_time < ?)", (cutoff_date,))
            cursor.execute("DELETE FROM pipeline_runs WHERE start_time < ?", (cutoff_date,))
            
            # Clean up orphaned lineage relationships
            cursor.execute("""
                DELETE FROM lineage_relationships 
                WHERE created_at < ? 
                AND source_entity_id NOT IN (SELECT id FROM datasets UNION SELECT id FROM models)
            """, (cutoff_date,))
            
            conn.commit()
            self.logger.info(f"Cleaned up metadata older than {retention_days} days")


class DataLineageTracker:
    """
    Track data lineage throughout the pipeline execution
    """
    
    def __init__(self, metadata_store: MetadataStore):
        """
        Initialize lineage tracker
        
        Args:
            metadata_store: Metadata store instance
        """
        self.metadata_store = metadata_store
        self.logger = logging.getLogger(__name__)
    
    def generate_dataset_id(self, file_path: str) -> str:
        """Generate deterministic ID for dataset based on path and content"""
        # Use file path and modification time for ID generation
        file_path = Path(file_path)
        if file_path.exists():
            stat = file_path.stat()
            content_hash = hashlib.md5(f"{file_path}_{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()
        else:
            content_hash = hashlib.md5(str(file_path).encode()).hexdigest()
        
        return f"dataset_{content_hash[:16]}"
    
    def calculate_schema_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of dataset schema"""
        schema_info = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': df.shape
        }
        schema_str = json.dumps(schema_info, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def track_dataset(self, file_path: str, df: pd.DataFrame, 
                     quality_score: float = 0.0) -> str:
        """Track dataset in lineage"""
        dataset_id = self.generate_dataset_id(file_path)
        file_path_obj = Path(file_path)
        
        dataset_metadata = DatasetMetadata(
            id=dataset_id,
            name=file_path_obj.name,
            path=str(file_path),
            schema_hash=self.calculate_schema_hash(df),
            row_count=len(df),
            column_count=len(df.columns),
            file_size_bytes=file_path_obj.stat().st_size if file_path_obj.exists() else 0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            quality_score=quality_score,
            column_info={col: str(dtype) for col, dtype in df.dtypes.items()},
            statistics=df.describe(include='all').to_dict() if not df.empty else {}
        )
        
        self.metadata_store.store_dataset_metadata(dataset_metadata)
        return dataset_id
    
    def track_transformation(self, input_dataset_id: str, output_dataset_id: str,
                           transformation_type: str, metadata: Dict[str, Any] = None):
        """Track transformation between datasets"""
        relationship_id = str(uuid.uuid4())
        
        relationship = LineageRelationship(
            id=relationship_id,
            source_entity_id=input_dataset_id,
            target_entity_id=output_dataset_id,
            relationship_type=LineageType.TRANSFORMED_TO,
            created_at=datetime.now(),
            metadata={
                'transformation_type': transformation_type,
                **(metadata or {})
            }
        )
        
        self.metadata_store.store_lineage_relationship(relationship)
        return relationship_id
    
    def track_model_training(self, model_id: str, training_dataset_id: str):
        """Track model training relationship"""
        relationship_id = str(uuid.uuid4())
        
        relationship = LineageRelationship(
            id=relationship_id,
            source_entity_id=training_dataset_id,
            target_entity_id=model_id,
            relationship_type=LineageType.USED_BY,
            created_at=datetime.now(),
            metadata={'relationship': 'training'}
        )
        
        self.metadata_store.store_lineage_relationship(relationship)
        return relationship_id


# Global metadata store instance
_metadata_store: Optional[MetadataStore] = None

def get_metadata_store(db_path: str = "metadata/pipeline_metadata.db") -> MetadataStore:
    """Get global metadata store instance"""
    global _metadata_store
    
    if _metadata_store is None:
        _metadata_store = MetadataStore(db_path)
    
    return _metadata_store
