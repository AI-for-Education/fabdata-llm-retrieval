from .datastore import DataStore
import os


async def get_datastore(**client_kwargs) -> DataStore:
    datastore = os.environ.get("DATASTORE")
    assert datastore is not None

    match datastore:
        case "redis":
            from .providers.redis_datastore import RedisDataStore

            return await RedisDataStore.init(**client_kwargs)
        case _:
            raise ValueError(f"Unsupported vector database: {datastore}")
