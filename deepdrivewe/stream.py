from typing import Any

from deepdrivewe.api import BaseModel
from proxystore.connectors.redis import RedisConnector
from proxystore.store import Store, get_store, register_store
from proxystore.stream.interface import StreamConsumer, StreamProducer
from proxystore.stream.shims.redis import RedisQueueSubscriber
from proxystore.stream.shims.redis import RedisQueuePublisher


class ProxyStreamConfig(BaseModel):
    store_name: str = 'sim-stream'
    redis_host: str = 'localhost'
    redis_port: int = 6379

    def get_store(self) -> Store[Any]:
        store = get_store(self.store_name)
        if store is None:
            connector = RedisConnector(self.redis_host, self.redis_port)
            store = Store(self.store_name, connector)
            register_store(store)
        return store

    # The StreamConsumer is generic on the type of the stream items.
    def get_consumer(self) -> StreamConsumer[Any]:
        subscriber = RedisQueueSubscriber(
            self.redis_host,
            self.redis_port,
            topic=self.store_name,
        )
        return StreamConsumer(subscriber)

    def get_producer(self) -> StreamProducer[Any]:
        store = self.get_store()
        publisher = RedisQueuePublisher(self.redis_host, self.redis_port)
        return StreamProducer(publisher, {self.store_name: store})
