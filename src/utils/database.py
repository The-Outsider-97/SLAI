
import psycopg2
import json

from datetime import datetime
from typing import Dict, Any

from logs.logger import get_logger

logger = get_logger("Database")

class IssueDBConnector:
    """PostgreSQL connector for safety issue tracking"""
    
    def __init__(self, host='localhost', port=5432, database='safety_issues',
                 user='admin', password='securepassword'):
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        self.cursor = self.conn.cursor()
        logger.info("Database connection established")

    def initialize_schema(self, tables: Dict[str, str]):
        """Create database tables from SQL definitions"""
        try:
            for table_name, ddl in tables.items():
                self.cursor.execute(ddl)
            self.conn.commit()
            logger.info(f"Created {len(tables)} tables")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Schema initialization failed: {e}")
            raise

    def log_issue(self, issue_data: Dict[str, Any]):
        """Insert a new safety issue record"""
        query = """
            INSERT INTO evaluation_issues 
            (id, timestamp, issue_type, severity, context, metrics)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        try:
            self.cursor.execute(query, (
                issue_data['id'],
                issue_data['timestamp'],
                issue_data['type'],
                issue_data['severity'],
                json.dumps(issue_data.get('context', {})),
                json.dumps(issue_data.get('metrics', {}))
            ))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to log issue: {e}")
            return False

class FallbackIssueTracker:
    """In-memory issue tracker for when database is unavailable"""
    
    def __init__(self):
        self.issues = []
        logger.warning("Using fallback issue tracking")

    def log_issue(self, issue_data: Dict[str, Any]):
        """Store issue in memory with timestamp"""
        self.issues.append({
            **issue_data,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def get_issues(self):
        """Retrieve all cached issues"""
        return self.issues.copy()
