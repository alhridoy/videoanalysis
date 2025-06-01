#!/usr/bin/env python3
"""Add metadata column to videos table if it doesn't exist"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text, inspect
from app.core.database import engine, Base
from app.models.video import Video, VideoFrame, ChatMessage

def add_metadata_column():
    """Add metadata column to videos table if it doesn't exist"""
    
    # Create inspector
    inspector = inspect(engine)
    
    # Check if videos table exists
    if 'videos' not in inspector.get_table_names():
        print("Videos table doesn't exist. Creating all tables...")
        Base.metadata.create_all(bind=engine)
        print("All tables created successfully!")
        return
    
    # Check if metadata column already exists
    columns = [col['name'] for col in inspector.get_columns('videos')]
    
    if 'metadata' in columns:
        print("Metadata column already exists in videos table.")
        return
    
    # Add metadata column
    print("Adding metadata column to videos table...")
    
    with engine.connect() as conn:
        # Use raw SQL to add the column
        conn.execute(text("ALTER TABLE videos ADD COLUMN metadata JSON"))
        conn.commit()
    
    print("Metadata column added successfully!")

if __name__ == "__main__":
    try:
        add_metadata_column()
    except Exception as e:
        print(f"Error adding metadata column: {e}")
        sys.exit(1)