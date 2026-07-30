"""Kafka producer wrapper for publishing filing messages.

References:
  - TDD: FR-3 (publish to filings.raw topic with metadata)
  - TDD: Section 5.2.1 Kafka message schema
  - TDD: Section 5.3 Kafka topics (filings.raw)
  - TDD: Section 8.1.1 (kafka publish counter)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog
from confluent_kafka import KafkaError, KafkaException, Producer

from src.config import settings
from src.metrics import FILING_SIZE_BYTES, KAFKA_PUBLISH_TOTAL

if TYPE_CHECKING:
    from src.edgar_client import Filing

logger = structlog.get_logger()

TOPIC_FILINGS_RAW = "filings.raw"

_KAFKA_MAX_MESSAGE_BYTES = 157_286_400  # 150 MB — must match broker KAFKA_MESSAGE_MAX_BYTES

# How long to wait for the broker to acknowledge one filing before treating it
# as undelivered.  Generous because acks=all on a multi-MB message is slow.
_DELIVERY_TIMEOUT_SECONDS = 120.0


class PublishOutcome(StrEnum):
    """Result of attempting to publish one filing."""

    DELIVERED = "delivered"       # broker acknowledged the message
    TOO_LARGE = "too_large"       # payload over MAX_RAW_BYTES, never produced
    FAILED = "failed"             # produce errored, was rejected, or timed out


def _delivery_callback(err: KafkaError | None, msg: Any) -> None:
    """Callback invoked on message delivery (or failure)."""
    if err is not None:
        KAFKA_PUBLISH_TOTAL.labels(topic=msg.topic(), status="error").inc()
        logger.error(
            "kafka_delivery_failed",
            topic=msg.topic(),
            error=str(err),
        )
    else:
        KAFKA_PUBLISH_TOTAL.labels(topic=msg.topic(), status="success").inc()
        logger.debug(
            "kafka_delivery_success",
            topic=msg.topic(),
            partition=msg.partition(),
            offset=msg.offset(),
        )


class FilingProducer:
    """Publishes Filing objects to the filings.raw Kafka topic.

    Message schema matches TDD Section 5.2.1:
    {accession_number, ticker, filing_date, filing_type,
     company_name, raw_text, source_url, published_at}
    """

    # Skip filings whose raw JSON exceeds this before compression.
    # At ~10:1 text compression ratio this keeps compressed messages well
    # under the broker's 20 MB limit.
    MAX_RAW_BYTES = 150 * 1024 * 1024  # 150 MB

    def __init__(self, bootstrap_servers: str | None = None) -> None:
        servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._producer = Producer(
            {
                "bootstrap.servers": servers,
                "client.id": "ingestion-service",
                "acks": "all",
                "compression.type": "gzip",
                "message.max.bytes": _KAFKA_MAX_MESSAGE_BYTES,
            }
        )

    def publish_filing(
        self,
        filing: Filing,
        *,
        timeout: float = _DELIVERY_TIMEOUT_SECONDS,
    ) -> PublishOutcome:
        """Serialize a Filing and publish it to Kafka, waiting for the ack (FR-3).

        ``produce()`` only enqueues into librdkafka's buffer; delivery happens
        later and can still fail.  The caller records the filing in
        ``ingestion_log``, which permanently suppresses re-ingestion of that
        accession number, so returning before the broker has acknowledged the
        message risks losing a filing with no way to notice.  This method
        therefore blocks until delivery is confirmed and reports the outcome.
        """
        message = {
            "accession_number": filing.accession_number,
            "ticker": filing.ticker,
            "filing_date": filing.filing_date,
            "filing_type": filing.filing_type,
            "company_name": filing.company_name,
            "raw_text": filing.raw_text,
            "source_url": filing.source_url,
            "published_at": datetime.now(UTC).isoformat(),
        }

        payload = json.dumps(message)
        raw_bytes = len(payload.encode())
        FILING_SIZE_BYTES.observe(raw_bytes)

        if raw_bytes > self.MAX_RAW_BYTES:
            KAFKA_PUBLISH_TOTAL.labels(topic=TOPIC_FILINGS_RAW, status="skipped_too_large").inc()
            logger.warning(
                "filing_skipped_too_large",
                ticker=filing.ticker,
                accession=filing.accession_number,
                raw_mb=round(raw_bytes / 1024 / 1024, 1),
                limit_mb=round(self.MAX_RAW_BYTES / 1024 / 1024, 1),
            )
            return PublishOutcome.TOO_LARGE

        # Captured by the per-message callback below; non-empty means the
        # broker rejected this specific message.
        delivery_errors: list[str] = []

        def _on_delivery(err: KafkaError | None, msg: Any) -> None:
            _delivery_callback(err, msg)
            if err is not None:
                delivery_errors.append(str(err))

        try:
            self._producer.produce(
                topic=TOPIC_FILINGS_RAW,
                key=filing.accession_number,
                value=payload,
                callback=_on_delivery,
            )
        except (BufferError, KafkaException) as exc:
            KAFKA_PUBLISH_TOTAL.labels(topic=TOPIC_FILINGS_RAW, status="error").inc()
            logger.error(
                "kafka_produce_failed",
                ticker=filing.ticker,
                accession=filing.accession_number,
                error=str(exc),
            )
            return PublishOutcome.FAILED

        # Block until this message is acknowledged. flush() returns the number
        # of messages still queued, so a non-zero result means we timed out.
        remaining = self._producer.flush(timeout)
        if remaining:
            logger.error(
                "kafka_delivery_timeout",
                ticker=filing.ticker,
                accession=filing.accession_number,
                timeout_seconds=timeout,
                still_queued=remaining,
            )
            return PublishOutcome.FAILED

        if delivery_errors:
            logger.error(
                "kafka_delivery_rejected",
                ticker=filing.ticker,
                accession=filing.accession_number,
                error=delivery_errors[0],
            )
            return PublishOutcome.FAILED

        return PublishOutcome.DELIVERED

    def flush(self, timeout: float = 10.0) -> int:
        """Wait for all in-flight messages to be delivered.

        Returns the number of messages still in the queue
        (0 means all delivered).
        """
        remaining: int = self._producer.flush(timeout)
        return remaining
