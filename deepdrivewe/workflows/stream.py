"""Streaming configuration for the DeepDriveWE workflow."""

from __future__ import annotations

from typing import Any

from parsl.addresses import address_by_hostname
from proxystore.store import get_store
from proxystore.store import register_store
from proxystore.store import Store
from proxystore.store.config import StoreConfig
from proxystore.stream import StreamConsumer
from proxystore.stream import StreamProducer
from proxystore.stream.shims.redis import RedisQueuePublisher
from proxystore.stream.shims.redis import RedisQueueSubscriber
from pydantic import field_validator

from deepdrivewe import BaseModel

SIMULATION_TOPIC = 'simulation-output'
TRAIN_TOPIC = 'train-output'


class ProxyStreamConfig(BaseModel):
    """Configuration for the proxy stream."""

    store_config: StoreConfig
    redis_host: str = 'localhost'
    redis_port: int = 6379

    @field_validator('redis_host')
    @classmethod
    def validate_redis_host(cls, value: str) -> str:
        """Validate the Redis host."""
        # Get the hostname if the address is 'hostname'
        if value == 'hostname':
            value = address_by_hostname()

        return value

    @field_validator('store_config')
    @classmethod
    def validate_store_config(cls, value: StoreConfig) -> StoreConfig:
        """Validate the store configuration."""
        if value.connector.kind == 'redis':
            hostname = value.connector.options.get('hostname')
            if hostname is None:
                raise ValueError(
                    'Hostname is required for Redis connector '
                    'in store configuration. Use "hostname" to use the '
                    'hostname of the current machine.',
                )
            # If the hostname is 'hostname', look up the hostname and set it
            if hostname == 'hostname':
                hostname = address_by_hostname()
                value.connector.options['hostname'] = hostname
        return value

    def get_store(self) -> Store[Any]:
        """Get the store for the proxy stream.

        Returns
        -------
        Store
            The store for the proxy stream.
        """
        store = get_store(self.store_config.name)
        if store is None:
            store = Store.from_config(self.store_config)
            register_store(store, exist_ok=True)
        return store

    # The StreamConsumer is generic on the type of the stream items.
    def get_consumer(self, topic: str) -> StreamConsumer[Any]:
        """Get a consumer for a given topic.

        Parameters
        ----------
        topic: str
            The topic to consume.

        Returns
        -------
        StreamConsumer
            The consumer for the given topic.
        """
        # The RedisQueueSubscriber is *not* a broadcasting stream. I.e., each
        # stream item will only be consumed by one subscriber (the subscriber
        # that wins the race). For multi-consumer support, see the
        # RedisSubscriber and RedisPublisher.
        subscriber = RedisQueueSubscriber(
            self.redis_host,
            self.redis_port,
            topic=topic,
        )
        return StreamConsumer(subscriber)

    def get_producer(self, topic: str) -> StreamProducer[Any]:
        """Get a producer for a given topic.

        Parameters
        ----------
        topic: str
            The topic to produce.

        Returns
        -------
        StreamProducer
            The producer for the given topic.
        """
        store = self.get_store()
        publisher = RedisQueuePublisher(self.redis_host, self.redis_port)
        return StreamProducer(publisher, stores={topic: store})
