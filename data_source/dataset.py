"""
data_source/dataset.py
----------------------
dataset helpers for:

1. Causal LM sliding windows
2. Supervised dialogue with answer-only loss masking
3. *Curriculum Batching integrated with PRM Heuristics*

Safe for:
- Large corpora
- Distributed training
- Mixed dataset training
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler

from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass(frozen=True)
class QualityFilterConfig:
    min_chars: int = 8
    min_alpha_ratio: float = 0.20
    max_symbol_ratio: float = 0.60
    max_repeated_line_fraction: float = 0.50
    min_unique_token_ratio: float = 0.10


@dataclass(frozen=True)
class KafkaConfig:
    backend: str = "confluent-kafka"
    bootstrap_servers: str = "localhost:9092"
    topic: str = ""
    group_id: str = "ftkai-train"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    poll_timeout_s: float = 1.0
    max_empty_polls: Optional[int] = None


class DataQualityFilter:
    """
    Conservative text-quality gate.

    Intentionally removes only obvious garbage: empty fragments, symbol soup,
    repeated spam lines, and extremely low-diversity text.
    """

    _whitespace_re = re.compile(r"\s+")
    _token_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(self, config: Optional[QualityFilterConfig] = None) -> None:
        self.config = config or QualityFilterConfig()

    def normalize(self, text: str) -> str:
        return self._whitespace_re.sub(" ", text).strip()

    def keep(self, text: str) -> bool:
        raw_text = text
        text = self.normalize(text)
        if len(text) < self.config.min_chars:
            return False

        chars = len(text)
        alpha_chars = sum(ch.isalpha() for ch in text)
        symbol_chars = sum(not ch.isalnum() and not ch.isspace() for ch in text)

        alpha_ratio = alpha_chars / chars
        symbol_ratio = symbol_chars / chars

        if alpha_ratio < self.config.min_alpha_ratio:
            return False

        if symbol_ratio > self.config.max_symbol_ratio:
            return False

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if lines:
            unique_lines = len(set(lines))
            repeated_line_fraction = 1.0 - (unique_lines / len(lines))
            if repeated_line_fraction > self.config.max_repeated_line_fraction:
                return False

        tokens = self._token_re.findall(text.lower())
        if tokens:
            unique_token_ratio = len(set(tokens)) / len(tokens)
            if unique_token_ratio < self.config.min_unique_token_ratio:
                return False

        return True


def _make_quality_filter(
    enabled: bool,
    quality_config: Optional[QualityFilterConfig],
) -> Optional[DataQualityFilter]:
    if not enabled:
        return None
    return DataQualityFilter(quality_config)


def _decode_message_payload(payload: Any) -> Optional[str]:
    if payload is None:
        return None
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="ignore")
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        return json.dumps(payload)
    return str(payload)


def parse_kafka_training_message(payload: Any) -> Optional[Dict[str, str]]:
    """
    Parse a Kafka message into the supervised training row shape.

    Expected message body:
      {"prompt": "...", "completion": "..."}

    Also accepts:
      {"question": "...", "final_answer": "..."}
    """
    decoded = _decode_message_payload(payload)
    if decoded is None:
        return None

    try:
        obj = json.loads(decoded)
    except json.JSONDecodeError:
        logger.warning("Skipping Kafka message with invalid JSON payload")
        return None

    if not isinstance(obj, dict):
        return None

    prompt = str(obj.get("prompt", obj.get("question", ""))).strip()
    completion = str(obj.get("completion", obj.get("final_answer", ""))).strip()

    if not prompt or not completion:
        return None

    return {"prompt": prompt, "completion": completion}


class _KafkaConsumerClient:
    def validate(self) -> None:
        raise NotImplementedError

    def poll(self, timeout_s: float) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _ConfluentKafkaConsumerClient(_KafkaConsumerClient):
    def __init__(self, config: KafkaConfig) -> None:
        try:
            from confluent_kafka import Consumer
        except ImportError as exc:  # pragma: no cover - exercised by runtime environment
            raise RuntimeError(
                "confluent-kafka is not installed. Install it or switch kafka_backend to kafka-python."
            ) from exc

        settings = {
            "bootstrap.servers": config.bootstrap_servers,
            "group.id": config.group_id,
            "auto.offset.reset": config.auto_offset_reset,
            "enable.auto.commit": config.enable_auto_commit,
        }
        self._consumer = Consumer(settings)
        self._topic = config.topic
        self._timeout_s = config.poll_timeout_s
        self._consumer.subscribe([config.topic])

    def validate(self) -> None:
        metadata = self._consumer.list_topics(
            topic=self._topic,
            timeout=max(1.0, self._timeout_s),
        )
        topic_metadata = metadata.topics.get(self._topic)
        if topic_metadata is None:
            raise RuntimeError(
                f"Kafka topic '{self._topic}' was not found in broker metadata."
            )
        if topic_metadata.error is not None:
            raise RuntimeError(
                f"Kafka topic '{self._topic}' is not ready: {topic_metadata.error}"
            )

    def poll(self, timeout_s: float) -> Any:
        message = self._consumer.poll(timeout_s)
        if message is None:
            return None
        if message.error():
            raise RuntimeError(f"Kafka consumer error: {message.error()}")
        return message.value()

    def close(self) -> None:
        self._consumer.close()


class _KafkaPythonConsumerClient(_KafkaConsumerClient):
    def __init__(self, config: KafkaConfig) -> None:
        try:
            from kafka import KafkaConsumer
        except ImportError as exc:  # pragma: no cover - exercised by runtime environment
            raise RuntimeError(
                "kafka-python is not installed. Install it or switch kafka_backend to confluent-kafka."
            ) from exc

        self._consumer = KafkaConsumer(
            config.topic,
            bootstrap_servers=config.bootstrap_servers,
            group_id=config.group_id,
            auto_offset_reset=config.auto_offset_reset,
            enable_auto_commit=config.enable_auto_commit,
        )
        self._topic = config.topic

    def validate(self) -> None:
        partitions = self._consumer.partitions_for_topic(self._topic)
        if partitions is None:
            raise RuntimeError(
                f"Kafka topic '{self._topic}' was not found or broker metadata could not be loaded."
            )

    def poll(self, timeout_s: float) -> Any:
        records = self._consumer.poll(timeout_ms=max(1, int(timeout_s * 1000)))
        for messages in records.values():
            if messages:
                return messages[0].value
        return None

    def close(self) -> None:
        self._consumer.close()


def build_kafka_consumer(config: KafkaConfig) -> _KafkaConsumerClient:
    if config.backend == "confluent-kafka":
        return _ConfluentKafkaConsumerClient(config)
    if config.backend == "kafka-python":
        return _KafkaPythonConsumerClient(config)
    raise ValueError(f"Unsupported Kafka backend: {config.backend}")


def validate_kafka_startup(
    config: KafkaConfig,
    consumer_factory: Optional[Callable[[KafkaConfig], _KafkaConsumerClient]] = None,
) -> None:
    consumer = (consumer_factory or build_kafka_consumer)(config)
    try:
        consumer.validate()
    finally:
        consumer.close()


# ============================================================
# Utilities
# ============================================================


def _read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    text = path.read_text(encoding="utf-8")
    return text.strip()


def _safe_load_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path} line {line_no}: {e}"
                ) from e

            if not isinstance(obj, dict):
                continue

            prompt = str(obj.get("prompt", "")).strip()
            completion = str(obj.get("completion", "")).strip()

            if prompt and completion:
                rows.append({"prompt": prompt, "completion": completion})

    return rows


def _iter_jsonl_rows(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path} line {line_no}: {e}"
                ) from e

            if not isinstance(obj, dict):
                continue

            prompt = str(obj.get("prompt", "")).strip()
            completion = str(obj.get("completion", "")).strip()

            if prompt and completion:
                yield {"prompt": prompt, "completion": completion}


def _looks_like_jsonl(text: str) -> bool:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            return False
        try:
            obj = json.loads(line)
        except Exception:
            return False
        return isinstance(obj, dict) and "prompt" in obj and "completion" in obj
    return False


# ============================================================
# Sliding Window Dataset (Causal LM)
# ============================================================


class SlidingWindowTextDataset(Dataset):
    """
    Memory-efficient sliding window causal LM dataset.
    """

    def __init__(
        self,
        text: str,
        tokenizer: Tokenizer,
        max_length: int,
        stride: int,
    ):
        if max_length < 2:
            raise ValueError("max_length must be >= 2")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        token_ids = tokenizer.encode(text)

        if len(token_ids) <= max_length:
            raise ValueError(
                f"Text too short ({len(token_ids)} tokens) for max_length={max_length}"
            )

        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.max_length = max_length
        self.stride = stride

        self.num_samples = ((len(self.tokens) - max_length - 1) // stride) + 1
        if self.num_samples <= 0:
            raise ValueError("No sliding windows could be created.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.max_length

        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]

        attention_mask = torch.ones_like(x)

        return {
            "input_ids": x,
            "labels": y,
            "attention_mask": attention_mask,
        }


class StreamingTextDataset(IterableDataset):
    """
    Streaming causal-LM dataset for line-based raw text corpora.
    """

    def __init__(
        self,
        data_paths: Iterable[str],
        tokenizer: Tokenizer,
        max_length: int,
        stride: int,
        quality_filter: Optional[DataQualityFilter] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.data_paths = tuple(data_paths)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.quality_filter = quality_filter
        self.rank = rank
        self.world_size = world_size

    def _iter_lines(self) -> Iterator[str]:
        for raw_path in self.data_paths:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if self.quality_filter is not None and not self.quality_filter.keep(line):
                        continue
                    yield line

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        shard_mod = max(1, self.world_size * num_workers)
        shard_index = self.rank * num_workers + worker_id

        token_buffer: List[int] = []
        sample_idx = 0

        for line in self._iter_lines():
            token_buffer.extend(self.tokenizer.encode(line))

            while len(token_buffer) > self.max_length:
                if sample_idx % shard_mod == shard_index:
                    x = torch.tensor(token_buffer[: self.max_length], dtype=torch.long)
                    y = torch.tensor(token_buffer[1 : self.max_length + 1], dtype=torch.long)
                    yield {
                        "input_ids": x,
                        "labels": y,
                        "attention_mask": torch.ones_like(x),
                    }

                del token_buffer[: self.stride]
                sample_idx += 1


class StreamingSupervisedDialogueDataset(IterableDataset):
    """
    Streaming prompt-completion dataset for jsonl sources.
    """

    def __init__(
        self,
        data_paths: Iterable[str],
        tokenizer: Tokenizer,
        max_length: int,
        quality_filter: Optional[DataQualityFilter] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.data_paths = tuple(data_paths)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.quality_filter = quality_filter
        self.rank = rank
        self.world_size = world_size

    def _iter_rows(self) -> Iterator[Dict[str, str]]:
        for raw_path in self.data_paths:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if path.suffix.lower() != ".jsonl":
                raise ValueError(
                    f"Streaming supervised mode supports .jsonl only, got: {path.name}"
                )

            for row in _iter_jsonl_rows(path):
                merged = f"{row['prompt']}\n{row['completion']}"
                if self.quality_filter is not None and not self.quality_filter.keep(merged):
                    continue
                yield row

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        shard_mod = max(1, self.world_size * num_workers)
        shard_index = self.rank * num_workers + worker_id

        for row_idx, row in enumerate(self._iter_rows()):
            if row_idx % shard_mod != shard_index:
                continue

            formatted = self.tokenizer.format_chat(
                system=None,
                user=row["prompt"],
                assistant=row["completion"],
            )

            if len(formatted) < 2:
                continue

            if len(formatted) > self.max_length + 1:
                formatted = formatted[: self.max_length + 1]

            input_ids = formatted[:-1]
            labels = formatted[1:]

            assistant_token_id = self.tokenizer.assistant_token_id
            mask_started = False
            for i, token_id in enumerate(input_ids):
                if token_id == assistant_token_id:
                    mask_started = True
                    continue
                if not mask_started:
                    labels[i] = IGNORE_INDEX

            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [self.tokenizer.pad_token_id] * pad_len
                labels += [IGNORE_INDEX] * pad_len

            attention_mask = [
                1 if i < len(formatted) - 1 else 0
                for i in range(self.max_length)
            ]

            yield {
                "input_ids": torch.tensor(input_ids[: self.max_length], dtype=torch.long),
                "labels": torch.tensor(labels[: self.max_length], dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }


class KafkaSupervisedDialogueDataset(IterableDataset):
    """
    Kafka-backed prompt-completion streaming dataset.

    Each Kafka message is expected to contain JSON with `prompt` and
    `completion` fields, or `question` and `final_answer`.
    """

    def __init__(
        self,
        kafka_config: KafkaConfig,
        tokenizer: Tokenizer,
        max_length: int,
        quality_filter: Optional[DataQualityFilter] = None,
        consumer_factory: Optional[Callable[[KafkaConfig], _KafkaConsumerClient]] = None,
    ) -> None:
        super().__init__()
        self.kafka_config = kafka_config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.quality_filter = quality_filter
        self.consumer_factory = consumer_factory or build_kafka_consumer

    def __iter__(self):
        consumer = self.consumer_factory(self.kafka_config)
        empty_polls = 0
        try:
            while True:
                payload = consumer.poll(self.kafka_config.poll_timeout_s)
                if payload is None:
                    empty_polls += 1
                    if (
                        self.kafka_config.max_empty_polls is not None and
                        empty_polls >= self.kafka_config.max_empty_polls
                    ):
                        break
                    continue

                empty_polls = 0
                row = parse_kafka_training_message(payload)
                if row is None:
                    continue

                merged = f"{row['prompt']}\n{row['completion']}"
                if self.quality_filter is not None and not self.quality_filter.keep(merged):
                    continue

                formatted = self.tokenizer.format_chat(
                    system=None,
                    user=row["prompt"],
                    assistant=row["completion"],
                )

                if len(formatted) < 2:
                    continue

                if len(formatted) > self.max_length + 1:
                    formatted = formatted[: self.max_length + 1]

                input_ids = formatted[:-1]
                labels = formatted[1:]

                assistant_token_id = self.tokenizer.assistant_token_id
                mask_started = False
                for i, token_id in enumerate(input_ids):
                    if token_id == assistant_token_id:
                        mask_started = True
                        continue
                    if not mask_started:
                        labels[i] = IGNORE_INDEX

                pad_len = self.max_length - len(input_ids)
                if pad_len > 0:
                    input_ids += [self.tokenizer.pad_token_id] * pad_len
                    labels += [IGNORE_INDEX] * pad_len

                attention_mask = [
                    1 if i < len(formatted) - 1 else 0
                    for i in range(self.max_length)
                ]

                yield {
                    "input_ids": torch.tensor(input_ids[: self.max_length], dtype=torch.long),
                    "labels": torch.tensor(labels[: self.max_length], dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
        finally:
            consumer.close()


# ============================================================
# Supervised Dialogue Dataset
# ============================================================


class SupervisedDialogueDataset(Dataset):
    """
    Prompt-completion dataset with answer-only loss masking.
    """

    def __init__(
        self,
        rows: List[Dict[str, str]],
        tokenizer: Tokenizer,
        max_length: int,
    ):
        if max_length < 8:
            raise ValueError("max_length too small for dialogue training.")

        self.pad_token_id = tokenizer.pad_token_id
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for row in rows:
            prompt = row["prompt"]
            completion = row["completion"]

            formatted = tokenizer.format_chat(
                system=None,
                user=prompt,
                assistant=completion,
            )

            if len(formatted) < 2:
                continue

            if len(formatted) > max_length + 1:
                formatted = formatted[: max_length + 1]

            input_ids = formatted[:-1]
            labels = formatted[1:]

            # Mask everything before assistant token
            assistant_token_id = tokenizer.assistant_token_id
            mask_started = False

            for i, token_id in enumerate(input_ids):
                if token_id == assistant_token_id:
                    mask_started = True
                    continue

                if not mask_started:
                    labels[i] = IGNORE_INDEX

            # Pad
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [self.pad_token_id] * pad_len
                labels += [IGNORE_INDEX] * pad_len

            attention_mask = [
                1 if i < len(formatted) - 1 else 0
                for i in range(max_length)
            ]

            self.samples.append(
                (
                    torch.tensor(input_ids[:max_length], dtype=torch.long),
                    torch.tensor(labels[:max_length], dtype=torch.long),
                    torch.tensor(attention_mask, dtype=torch.long),
                )
            )

        if not self.samples:
            raise ValueError("No valid supervised samples created.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        input_ids, labels, attention_mask = self.samples[idx]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# ============================================================
# Dataset Builder
# ============================================================


def build_training_dataset(
    data_paths: Iterable[str],
    tokenizer: Tokenizer,
    max_length: int,
    stride: int,
    use_curriculum: bool = False,
    use_quality_filter: bool = True,
    quality_config: Optional[QualityFilterConfig] = None,
) -> Tuple[Dataset, Optional[List[str]]]:
    """
    Returns (Dataset, Optional raw_texts for CurriculumScorer)
    """
    datasets: List[Dataset] = []
    all_raw_texts: Optional[List[str]] = [] if use_curriculum else None
    quality_filter = _make_quality_filter(use_quality_filter, quality_config)

    for raw_path in data_paths:
        path = Path(raw_path)
        print(f"DEBUG: Processing {path}")
        raw_text = _read_text_file(path)
        if not raw_text:
            print(f"DEBUG: {path} is empty")
            continue

        if path.suffix.lower() == ".json":
            # Structured JSON dataset
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if "entries" in data:
                rows = []
                for entry in data["entries"]:
                    prompt = str(entry.get("question", "")).strip()
                    completion = str(entry.get("final_answer", "")).strip()
                    if prompt and completion:
                        merged = f"{prompt}\n{completion}"
                        if quality_filter is not None and not quality_filter.keep(merged):
                            continue
                        rows.append({
                            "prompt": prompt,
                            "completion": completion,
                        })

                if rows:
                    datasets.append(
                        SupervisedDialogueDataset(
                            rows,
                            tokenizer,
                            max_length,
                        )
                    )
                    if all_raw_texts is not None:
                        all_raw_texts.extend([r["prompt"] + "\n" + r["completion"] for r in rows])
                else:
                    print(f"DEBUG: {path} had no valid rows after filtering")
            else:
                raise ValueError("JSON file must contain 'entries' key.")

        elif path.suffix.lower() == ".jsonl" or _looks_like_jsonl(raw_text):
            rows = _safe_load_jsonl(path)
            if quality_filter is not None:
                rows = [
                    row for row in rows
                    if quality_filter.keep(f"{row['prompt']}\n{row['completion']}")
                ]
            if rows:
                datasets.append(
                    SupervisedDialogueDataset(
                        rows,
                        tokenizer,
                        max_length,
                    )
                )
                if all_raw_texts is not None:
                    all_raw_texts.extend([r["prompt"] + "\n" + r["completion"] for r in rows])
            else:
                print(f"DEBUG: {path} (jsonl) had no valid rows after filtering")
        else:
            if quality_filter is not None and not quality_filter.keep(raw_text):
                print(f"DEBUG: {path} failed quality filter")
                continue
            ds = SlidingWindowTextDataset(
                raw_text,
                tokenizer,
                max_length,
                stride,
            )
            datasets.append(ds)
            if all_raw_texts is not None:
                # Sliding window doesn't naturally support curriculum scoring well, 
                # assign generic empty strings so lengths match Dataset indices.
                all_raw_texts.extend(["" for _ in range(len(ds))])

    if not datasets:
        raise ValueError("No valid datasets constructed.")

    final_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return final_dataset, all_raw_texts


def build_streaming_dataset(
    data_paths: Iterable[str],
    tokenizer: Tokenizer,
    max_length: int,
    stride: int,
    rank: int = 0,
    world_size: int = 1,
    use_quality_filter: bool = True,
    quality_config: Optional[QualityFilterConfig] = None,
) -> IterableDataset:
    quality_filter = _make_quality_filter(use_quality_filter, quality_config)
    paths = tuple(data_paths)
    if not paths:
        raise ValueError("data_paths must be non-empty")

    # Real Dataset Streaming Pipeline (HuggingFace datasets)
    if any(p.startswith("hf://") for p in paths):
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("pip install datasets to stream from HuggingFace")
        
        class HuggingFaceStreamingDataset(IterableDataset):
            def __init__(self, hf_paths, tokenizer, max_length, rank, world_size):
                super().__init__()
                self.hf_paths = hf_paths
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.rank = rank
                self.world_size = world_size
                
            def __iter__(self):
                worker = get_worker_info()
                worker_id = worker.id if worker is not None else 0
                num_workers = worker.num_workers if worker is not None else 1
                shard_mod = max(1, self.world_size * num_workers)
                shard_index = self.rank * num_workers + worker_id

                for path in self.hf_paths:
                    dataset_name = path[5:]  # strip hf://
                    ds = load_dataset(dataset_name, split="train", streaming=True)
                    
                    token_buffer = []
                    sample_idx = 0
                    
                    for row in ds:
                        text = row.get("text", row.get("content", ""))
                        if not text: continue
                        
                        token_buffer.extend(self.tokenizer.encode(text))
                        
                        while len(token_buffer) > self.max_length:
                            if sample_idx % shard_mod == shard_index:
                                x = torch.tensor(token_buffer[: self.max_length], dtype=torch.long)
                                y = torch.tensor(token_buffer[1 : self.max_length + 1], dtype=torch.long)
                                yield {
                                    "input_ids": x,
                                    "labels": y,
                                    "attention_mask": torch.ones_like(x),
                                }
                            del token_buffer[: self.max_length]  # chunk without stride to simulate standard block processing
                            sample_idx += 1

        return HuggingFaceStreamingDataset(paths, tokenizer, max_length, rank, world_size)

    suffixes = {Path(path).suffix.lower() for path in paths}
    if ".json" in suffixes:
        raise ValueError(
            "Streaming mode does not support top-level .json datasets. Convert them to .jsonl first."
        )

    if suffixes == {".jsonl"}:
        return StreamingSupervisedDialogueDataset(
            data_paths=paths,
            tokenizer=tokenizer,
            max_length=max_length,
            quality_filter=quality_filter,
            rank=rank,
            world_size=world_size,
        )

    if ".jsonl" in suffixes and len(suffixes) > 1:
        raise ValueError(
            "Streaming mode does not support mixing .jsonl dialogue files with raw text files in one loader."
        )

    return StreamingTextDataset(
        data_paths=paths,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        quality_filter=quality_filter,
        rank=rank,
        world_size=world_size,
    )


def build_kafka_streaming_dataset(
    kafka_config: KafkaConfig,
    tokenizer: Tokenizer,
    max_length: int,
    use_quality_filter: bool = True,
    quality_config: Optional[QualityFilterConfig] = None,
    consumer_factory: Optional[Callable[[KafkaConfig], _KafkaConsumerClient]] = None,
) -> IterableDataset:
    quality_filter = _make_quality_filter(use_quality_filter, quality_config)
    return KafkaSupervisedDialogueDataset(
        kafka_config=kafka_config,
        tokenizer=tokenizer,
        max_length=max_length,
        quality_filter=quality_filter,
        consumer_factory=consumer_factory,
    )


# ============================================================
# Dataloader Builder
# ============================================================


def create_dataloader_from_paths(
    data_paths: Iterable[str],
    tokenizer: Tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    use_curriculum: bool = False,
    total_steps: int = 10000,
    curriculum_difficulty_window: float = 0.20,
    curriculum_easy_retention: float = 0.10,
    drop_last: bool = True,
    persistent_workers: bool = True,
    use_streaming: bool = False,
    use_kafka: bool = False,
    kafka_config: Optional[KafkaConfig] = None,
    use_quality_filter: bool = True,
    quality_config: Optional[QualityFilterConfig] = None,
    kafka_consumer_factory: Optional[Callable[[KafkaConfig], _KafkaConsumerClient]] = None,
) -> DataLoader:
    if use_kafka and use_curriculum:
        raise ValueError("Curriculum batching is not supported with Kafka datasets.")

    if use_streaming and use_curriculum:
        raise ValueError("Curriculum batching is not supported with streaming datasets.")

    if use_kafka:
        if kafka_config is None:
            raise ValueError("kafka_config must be provided when use_kafka=True")
        dataset = build_kafka_streaming_dataset(
            kafka_config=kafka_config,
            tokenizer=tokenizer,
            max_length=max_length,
            use_quality_filter=use_quality_filter,
            quality_config=quality_config,
            consumer_factory=kafka_consumer_factory,
        )
        raw_texts = None
    elif use_streaming:
        dataset = build_streaming_dataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride,
            rank=rank if distributed else 0,
            world_size=world_size if distributed else 1,
            use_quality_filter=use_quality_filter,
            quality_config=quality_config,
        )
        raw_texts = None
    else:
        dataset, raw_texts = build_training_dataset(
            data_paths,
            tokenizer,
            max_length,
            stride,
            use_curriculum=use_curriculum,
            use_quality_filter=use_quality_filter,
            quality_config=quality_config,
        )

    sampler: Optional[torch.utils.data.Sampler] = None  # type: ignore[assignment]
    batch_sampler: Optional[torch.utils.data.BatchSampler] = None  # type: ignore[assignment]

    if use_curriculum and raw_texts:
        try:
            import sys
            import os
            # Ensure PRM logic can be imported natively
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from PRM.curriculum import (
                CurriculumScheduler, 
                CurriculumSampler, 
                CurriculumDataset, 
                CurriculumConfig,
                HeuristicCodeScorer,
                ScoredExample
            )

            # Heuristic Soft Filtering + Scoring
            logger.info("Initializing PRM Quality Filter and Curriculum Builder...")
            scorer = HeuristicCodeScorer()
            scored_examples = []
            
            # Since PyTorch dataset indices must match perfectly, the generic "example" represents the int index
            for idx, txt in enumerate(raw_texts):
                axes = scorer.score(txt)
                scored_examples.append(ScoredExample(example=idx, difficulty=axes))
            
            curr_config = CurriculumConfig(
                difficulty_window=curriculum_difficulty_window,
                easy_retention_frac=curriculum_easy_retention,
            )
            
            scheduler = CurriculumScheduler(scored_examples, config=curr_config)
            curr_dataset = CurriculumDataset(scored_examples)
            
            # The CurriculumSampler acts as the batch_sampler natively iterating step arrays 
            batch_sampler = CurriculumSampler(  # type: ignore[assignment]
                dataset=curr_dataset,
                scheduler=scheduler,
                batch_size=batch_size,
                total_steps=total_steps,
                rank=rank if distributed else 0,
                world_size=world_size if distributed else 1,
            )
            
            shuffle = False
            sampler = None # Batch sampler fully replaces regular sampler logic

        except ImportError as e:
            logger.warning(f"Could not load PRM logic: {e}. Falling back to default dataloader.")
            use_curriculum = False

    if not use_curriculum and not use_streaming and not use_kafka:
        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            shuffle = False

    if isinstance(dataset, IterableDataset):
        shuffle = False
        sampler = None

    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers and num_workers > 0,
        )
