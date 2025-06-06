from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

import sys # sys is used by fileConfig
import os # os is used for path manipulation

# The `prepend_sys_path = .` in alembic.ini should handle adding the project root.
# Removing the manual sys.path.insert here to avoid potential conflicts or redundancy.
# # Add project root to Python path to allow finding crypto_trading module
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from crypto_trading.database.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata

# --- Customization to load DB URL from central_config.json ---
import json

# Assuming env.py is in alembic/ and config/ is a sibling to alembic/ at project root
# The existing sys.path.insert for '..' , '..' was confusing.
# If alembic.ini has prepend_sys_path = ., then project root is already in sys.path.
# And Base = declarative_base() from crypto_trading.database.models implies models are found.
# Path to central_config.json, assuming alembic/ is directly under project root
# so os.path.dirname(__file__) is /path/to/project_root/alembic
# then '..' goes to /path/to/project_root/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CENTRAL_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'central_config.json')

def get_db_url_from_central_config():
    try:
        with open(CENTRAL_CONFIG_PATH, 'r') as f:
            central_cfg = json.load(f)

        db_config = central_cfg.get('database')
        if not db_config:
            raise ValueError("Database configuration missing in central_config.json")

        db_type = db_config.get('type')
        db_name = db_config.get('name') # For SQLite, this is the relative path to the db file

        if db_type == 'sqlite':
            # Ensure the path to sqlite db is absolute or relative to project root
            # The 'name' in central_config.json is like "./BINANCE_TRADES.db"
            # If db_name starts with './', make it relative to project root.
            if db_name.startswith('./'):
                db_path = os.path.join(PROJECT_ROOT, db_name[2:])
            else: # Assume absolute path or a name to be placed in project root
                db_path = os.path.join(PROJECT_ROOT, db_name) # Fallback, might need adjustment

            # Ensure path is absolute for SQLAlchemy URL
            db_path = os.path.abspath(db_path)
            return f"sqlite:///{db_path}"
        # Add other database types here if needed (e.g., postgresql)
        elif db_type == 'postgresql':
            user = db_config.get('username')
            password = db_config.get('password')
            host = db_config.get('host')
            port = db_config.get('port')
            dbname = db_config.get('name')
            return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        else:
            raise ValueError(f"Unsupported database type: {db_type} in central_config.json")

    except Exception as e:
        # Fallback or error if central config can't be read or is invalid
        print(f"Error loading DB URL from {CENTRAL_CONFIG_PATH}: {e}")
        print("Falling back to sqlalchemy.url from alembic.ini if defined, or will error.")
        return None

target_metadata = Base.metadata
# --- End Customization ---

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # --- Customization: Inject DB URL from central_config.json ---
    loaded_db_url = get_db_url_from_central_config()
    if loaded_db_url:
        # Override the sqlalchemy.url from alembic.ini with the one from central_config.json
        # config.set_main_option('sqlalchemy.url', loaded_db_url) # This is one way

        # Another way: directly create engine configuration for engine_from_config
        engine_config = config.get_section(config.config_ini_section, {})
        engine_config['sqlalchemy.url'] = loaded_db_url # Override URL
    else:
        # If central config loading failed, proceed with URL from alembic.ini (if any)
        engine_config = config.get_section(config.config_ini_section, {})
        if not engine_config.get('sqlalchemy.url'):
            raise ValueError("Database URL could not be determined from central_config.json or alembic.ini")

    connectable = engine_from_config(
        engine_config, # Use potentially modified engine_config
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    # --- End Customization ---

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
