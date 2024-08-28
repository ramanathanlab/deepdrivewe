from typing import Any

from deepdrivewe.api import BaseModel
from proxystore.connectors.redis import RedisConnector
from proxystore.store.config import StoreConfig
from proxystore.store import Store, get_store, register_store
from proxystore.stream.interface import StreamConsumer, StreamProducer
from proxystore.stream.shims.redis import RedisQueueSubscriber
from proxystore.stream.shims.redis import RedisQueuePublisher


class ProxyStreamConfig(BaseModel):
    store_config: StoreConfig
    stream_topic: str = 'sim-stream'
    redis_host: str = 'localhost'
    redis_port: int = 6379

    def get_store(self) -> Store[Any]:
        store = get_store(self.store_config.name)
        if store is None:
            store = Store.from_config(self.store_config)
            register_store(store, exist_ok=True)
        return store

    # The StreamConsumer is generic on the type of the stream items.
    def get_consumer(self) -> StreamConsumer[Any]:
        # The RedisQueueSubscriber is *not* a broadcasting stream. I.e., each
        # stream item will only be consumed by one subscriber (the subscriber
        # that wins the race). For multi-consumer support, see the
        # RedisSubscriber and RedisPublisher.
        subscriber = RedisQueueSubscriber(
            self.redis_host,
            self.redis_port,
            topic=self.stream_topic,
        )
        return StreamConsumer(subscriber)

    def get_producer(self) -> StreamProducer[Any]:
        store = self.get_store()
        publisher = RedisQueuePublisher(self.redis_host, self.redis_port)
        return StreamProducer(publisher, {self.stream_topic: store})
