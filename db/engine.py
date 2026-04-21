from sqlalchemy import Engine, create_engine

from db.models import metadata


def make_engine(database_url: str) -> Engine:
    kwargs: dict = {}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(database_url, **kwargs)


def create_tables(engine: Engine) -> None:
    metadata.create_all(engine)
