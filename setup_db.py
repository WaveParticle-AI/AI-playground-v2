"""
Database setup verification script for Waveparticle Playground Evaluation System.
Run this after setting up your Neon database to verify connection.
"""

from playground.eval_db import get_db_connection, init_db, check_db_available
import sys

def main():
    print("=" * 60)
    print("Waveparticle Playground - Database Setup Check")
    print("=" * 60)
    
    # Check if psycopg2 is available
    try:
        import psycopg2
        print("✓ psycopg2-binary is installed")
    except ImportError:
        print("✗ psycopg2-binary not found. Run: pip install psycopg2-binary")
        sys.exit(1)
    
    # Check database connection
    print("\nChecking database connection...")
    if check_db_available():
        print("✓ Database connection successful!")
        
        # Initialize database
        print("\nInitializing database tables...")
        if init_db():
            print("✓ Database initialized successfully!")
            print("  Table 'evaluations' is ready.")
        else:
            print("✗ Failed to initialize database tables")
            sys.exit(1)
    else:
        print("✗ Database connection failed!")
        print("\nTo fix:")
        print("1. Sign up at https://neon.tech (free, no credit card)")
        print("2. Create a project named 'wave-particle-playground'")
        print("3. Copy the connection string from the dashboard")
        print("4. Add to .streamlit/secrets.toml:")
        print('   [database]')
        print('   url = "postgresql://user:pass@your-endpoint.neon.tech/..."')
        print("   OR set environment variable:")
        print("   export DATABASE_URL='postgresql://...'")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Setup complete! Your evaluation data will be saved to the cloud.")
    print("=" * 60)

if __name__ == "__main__":
    main()
