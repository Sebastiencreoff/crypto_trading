#!/usr/bin/env python
# coding: utf-8

import logging
import os
# Import the new config loader and schema
from config_management.loader import load_config
from config_management.schemas import AppConfig

# Global variable to hold the loaded configuration
# This will be an instance of AppConfig after initialization
app_config: AppConfig = None

def get_engine():
    """
    Provides a SQLAlchemy engine based on the loaded configuration.
    This function will need to be adapted or moved depending on how database
    connections are managed with the new AppConfig.
    For now, it's a placeholder or needs to be adapted if still used directly.
    """
    if app_config is None:
        raise RuntimeError("Configuration not initialized. Call config.init() first.")

    # Example: Construct SQLAlchemy URL from app_config.database
    # This is a simplified example and might need adjustment based on actual DB types and needs
    from sqlalchemy import create_engine
    db_conf = app_config.database
    if db_conf.type == "sqlite":
        # Assuming db_conf.name is a relative path from base_config_path or an absolute path
        db_path = db_conf.name
        if app_config.base_config_path and not os.path.isabs(db_path):
            db_path = os.path.join(app_config.base_config_path, db_path)

        # Ensure the directory for the SQLite DB exists
        db_dir = os.path.dirname(db_path)
        if db_dir: # Create directory if it's not the current directory and doesn't exist
            os.makedirs(db_dir, exist_ok=True)

        sqlalchemy_url = f"sqlite:///{db_path}"
    elif db_conf.type == "postgresql": # Example for other DBs
        sqlalchemy_url = f"postgresql://{db_conf.username}:{db_conf.password}@{db_conf.host}:{db_conf.port}/{db_conf.name}"
    else:
        raise ValueError(f"Unsupported database type: {db_conf.type}")

    logging.info(f"Creating SQLAlchemy engine for URL: {sqlalchemy_url}")
    return create_engine(sqlalchemy_url)

def get_session():
    """
    Provides a SQLAlchemy session.
    Similar to get_engine, this will rely on the initialized app_config.
    """
    # This is a simplified example. Session factory management might be needed.
    from sqlalchemy.orm import sessionmaker
    engine = get_engine() # Relies on the new get_engine()
    session_factory = sessionmaker(bind=engine)
    logging.info("SQLAlchemy session factory created.")
    return session_factory()

def init(config_file_path: str) -> AppConfig:
    """
    Initializes the application configuration using the new config management system.
    Loads the configuration from the given path, validates it, and stores it
    in the global `app_config` variable.
    """
    global app_config
    logging.info(f"Initializing configuration from: {config_file_path} using new config management.")

    try:
        app_config = load_config(config_file_path)
        logging.info(f"Configuration loaded successfully for service: {app_config.service_name}")

        # Example: Test database connection if configured
        if app_config.database:
            try:
                engine = get_engine()
                with engine.connect() as connection:
                    logging.info("Database connection successful via SQLAlchemy engine.")
            except Exception as e:
                logging.error(f"Failed to connect to database {app_config.database.name} using SQLAlchemy engine: {e}", exc_info=True)
                # Depending on requirements, you might want to raise e here
                # or allow the application to continue if DB is not critical at init.
    except Exception as e:
        logging.error(f"Failed to initialize application configuration: {e}", exc_info=True)
        raise # Re-raise the exception to prevent application from starting in a bad state

    return app_config
