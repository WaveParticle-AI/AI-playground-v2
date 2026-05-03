"""
PostgreSQL database module for persisting evaluation data.
Uses Neon Free Tier (or any PostgreSQL instance).
Falls back to session state if DB connection is unavailable.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Optional, List, Dict
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from playground.evaluation import SessionEvaluation, TurnEvaluation


def get_db_connection():
    """Get PostgreSQL connection from Streamlit secrets or environment variable."""
    if not PSYCOPG2_AVAILABLE:
        return None
    
    # Try Streamlit secrets first (for deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'database' in st.secrets:
            conn = psycopg2.connect(
                st.secrets['database']['url'],
                cursor_factory=RealDictCursor
            )
            return conn
    except Exception:
        pass
    
    # Try environment variable
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        try:
            conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
            return conn
        except Exception:
            pass
    
    return None


def init_db(conn=None) -> bool:
    """Create evaluations table if it doesn't exist. Returns True if DB is available."""
    if conn is None:
        conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id VARCHAR NOT NULL,
                    tenant_id VARCHAR DEFAULT 'default',
                    character_id VARCHAR NOT NULL,
                    model_name VARCHAR NOT NULL,
                    turn_num INT NOT NULL,
                    dim_scores JSONB,
                    pattern_hits JSONB,
                    auto_checks JSONB,
                    total_score INT,
                    user_message TEXT,
                    assistant_reply TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_evaluations_session 
                ON evaluations(session_id);
                CREATE INDEX IF NOT EXISTS idx_evaluations_character 
                ON evaluations(character_id);
            """)
        conn.commit()
        return True
    except Exception as e:
        print(f"DB init error: {e}")
        return False
    finally:
        if conn:
            conn.close()


def save_turn_evaluation(turn_eval: TurnEvaluation, session_id: str, 
                         tenant_id: str = "default") -> bool:
    """Persist a single turn evaluation to PostgreSQL."""
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO evaluations 
                (session_id, tenant_id, character_id, model_name, turn_num,
                 dim_scores, pattern_hits, auto_checks, total_score,
                 user_message, assistant_reply)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                session_id,
                tenant_id,
                turn_eval.character_id,
                turn_eval.model_name,
                turn_eval.turn_num,
                json.dumps(turn_eval.dim_scores),
                json.dumps(turn_eval.pattern_hits),
                json.dumps(turn_eval.auto_checks),
                turn_eval.total_score,
                turn_eval.user_message,
                turn_eval.assistant_reply
            ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Save error: {e}")
        return False
    finally:
        conn.close()


def save_session_evaluation(session_eval: SessionEvaluation) -> bool:
    """Persist all turn evaluations in a session."""
    success = True
    for turn_eval in session_eval.turn_evals:
        if not save_turn_evaluation(turn_eval, session_eval.session_id, session_eval.tenant_id):
            success = False
    return success


def load_session_evaluations(session_id: str) -> List[Dict]:
    """Load all evaluations for a session."""
    conn = get_db_connection()
    if conn is None:
        return []
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM evaluations 
                WHERE session_id = %s 
                ORDER BY turn_num ASC
            """, (session_id,))
            return cur.fetchall()
    except Exception as e:
        print(f"Load error: {e}")
        return []
    finally:
        conn.close()


def get_character_stats(character_id: str) -> Dict:
    """Get aggregated stats for a character across all sessions."""
    conn = get_db_connection()
    if conn is None:
        return {}
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_turns,
                    AVG(total_score) as avg_score,
                    character_id
                FROM evaluations 
                WHERE character_id = %s
                GROUP BY character_id
            """, (character_id,))
            return cur.fetchone() or {}
    except Exception as e:
        print(f"Stats error: {e}")
        return {}
    finally:
        conn.close()


def check_db_available() -> bool:
    """Check if database is available."""
    conn = get_db_connection()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False
    finally:
        conn.close()
