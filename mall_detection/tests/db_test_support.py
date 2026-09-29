"""Isolated SQLite setup shared by the application-level unittest modules."""

import os
import tempfile


_db_path = os.path.join(tempfile.gettempdir(), f"accutrack-tests-{os.getpid()}.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path.replace(os.sep, '/')}"
_initialized = False


def initialize_test_database(app_module):
    """Initialize the same file-backed schema used by the app's DB worker."""
    global _initialized
    if _initialized:
        return

    app_module.init_db(app_module.VIDEO_SOURCES, app_module.ZONES)
    # Verify through a fresh SQLAlchemy connection, as the background worker
    # uses its own thread-scoped session and therefore its own pooled checkout.
    from models import engine
    with engine.connect() as connection:
        connection.exec_driver_sql("SELECT id FROM cameras LIMIT 1").first()
    _initialized = True
