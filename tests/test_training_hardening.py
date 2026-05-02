from pathlib import Path

import torch

from checkpoint.checkpoint import load_checkpoint, latest_checkpoint, save_checkpoint
from config.model_config import ModelConfig
from config.train_config import TrainConfig
from data_source.dataset import (
    DataQualityFilter,
    KafkaConfig,
    QualityFilterConfig,
    create_dataloader_from_paths,
    parse_kafka_training_message,
    validate_kafka_startup,
)
from model.model import ModernLLM, build_attention_mask
from scheduler.cosine import get_scheduled_lr


class FakeTokenizer:
    pad_token_id = 0
    assistant_token_id = 99

    def encode(self, text: str):
        return [((ord(ch) - 31) % 31) + 1 for ch in text]

    def format_chat(self, system, user, assistant=None):
        tokens = []
        if system:
            tokens.extend(self.encode(system))
        tokens.extend(self.encode(user))
        if assistant:
            tokens.append(self.assistant_token_id)
            tokens.extend(self.encode(assistant))
        return tokens


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        context_length=8,
        emb_dim=16,
        hidden_dim=64,
        multiple_of=16,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        drop_rate=0.0,
        rope_scaling_factor=1.0,
    )


def test_modernllm_attention_mask_matches_explicit_mask():
    torch.manual_seed(0)
    model = ModernLLM(tiny_model_config())
    model.eval()

    tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    explicit_mask = build_attention_mask(
        attention_mask=attention_mask,
        query_len=tokens.size(1),
        key_len=tokens.size(1),
        dtype=torch.float32,
        device=tokens.device,
    )

    with torch.no_grad():
        masked_logits = model(tokens, attention_mask=attention_mask)["logits"]
        explicit_logits = model(tokens, mask=explicit_mask)["logits"]

    torch.testing.assert_close(masked_logits, explicit_logits, atol=1e-6, rtol=0.0)


def test_dataloader_respects_drop_last_and_persistent_workers(tmp_path):
    data_path = tmp_path / "train.txt"
    data_path.write_text("abcdefgh", encoding="utf-8")

    loader_keep = create_dataloader_from_paths(
        data_paths=(str(data_path),),
        tokenizer=FakeTokenizer(),
        batch_size=2,
        max_length=3,
        stride=2,
        num_workers=0,
        drop_last=False,
        persistent_workers=True,
    )
    loader_drop = create_dataloader_from_paths(
        data_paths=(str(data_path),),
        tokenizer=FakeTokenizer(),
        batch_size=2,
        max_length=3,
        stride=2,
        num_workers=0,
        drop_last=True,
        persistent_workers=True,
    )

    assert len(loader_keep) == 2
    assert len(loader_drop) == 1
    assert loader_keep.persistent_workers is False


def test_checkpoint_resume_restores_weights_and_optimizer(tmp_path):
    torch.manual_seed(0)
    model = ModernLLM(tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_config = TrainConfig(
        checkpoint_dir=str(tmp_path),
        data_path="unused.txt",
        data_paths=("unused.txt",),
    )

    input_tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    labels = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(
        model(input_tokens)["logits"].reshape(-1, model.config.vocab_size),
        labels.reshape(-1),
    )
    loss.backward()
    optimizer.step()

    original_weight = model.tok_emb.weight.detach().clone()
    ckpt_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=None,
        step=7,
        epoch=2,
        loss=float(loss.item()),
        model_config=model.config,
        train_config=train_config,
        checkpoint_dir=str(tmp_path),
    )

    with torch.no_grad():
        model.tok_emb.weight.zero_()

    state = load_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        device=torch.device("cpu"),
        strict=True,
    )

    assert state["step"] == 7
    assert state["epoch"] == 2
    torch.testing.assert_close(model.tok_emb.weight, original_weight)
    assert latest_checkpoint(str(tmp_path)) == ckpt_path
    assert not any(path.name.startswith(".tmp_ckpt_") for path in Path(tmp_path).iterdir())


def test_train_config_and_scheduler_are_authoritative(tmp_path):
    cfg = TrainConfig(
        checkpoint_dir=str(tmp_path),
        data_path="train.txt",
        data_paths=("train.txt",),
        eval_data_path="eval.txt",
        optimizer="adamw",
        scheduler="linear",
        deterministic=True,
        drop_last=False,
        persistent_workers=False,
    )

    assert cfg.eval_data_paths == ("eval.txt",)
    assert cfg.drop_last is False
    assert cfg.persistent_workers is False
    assert cfg.deterministic is True
    assert get_scheduled_lr("constant", step=5, warmup_steps=0, total_steps=10, peak_lr=1e-3, min_lr=1e-4) == 1e-3
    assert get_scheduled_lr("linear", step=10, warmup_steps=0, total_steps=10, peak_lr=1e-3, min_lr=1e-4) == 1e-4


def test_quality_filter_is_conservative():
    quality_filter = DataQualityFilter(
        QualityFilterConfig(
            min_chars=8,
            min_alpha_ratio=0.20,
            max_symbol_ratio=0.60,
            max_repeated_line_fraction=0.50,
            min_unique_token_ratio=0.10,
        )
    )

    assert quality_filter.keep("Write a Python function that adds two numbers.")
    assert quality_filter.keep("User asks for a summary of a report and gets a concise answer.")
    assert not quality_filter.keep("!!!!@@@@####$$$$")
    assert not quality_filter.keep("spam\nspam\nspam\nspam")


def test_streaming_text_dataloader_yields_batches(tmp_path):
    data_path = tmp_path / "stream.txt"
    data_path.write_text(
        "Normal training text line one.\n"
        "!!!!@@@@####$$$$\n"
        "Another clean line for the model.\n",
        encoding="utf-8",
    )

    loader = create_dataloader_from_paths(
        data_paths=(str(data_path),),
        tokenizer=FakeTokenizer(),
        batch_size=1,
        max_length=4,
        stride=2,
        num_workers=0,
        use_streaming=True,
        use_quality_filter=True,
        quality_config=QualityFilterConfig(
            min_chars=8,
            min_alpha_ratio=0.20,
            max_symbol_ratio=0.60,
            max_repeated_line_fraction=0.50,
            min_unique_token_ratio=0.10,
        ),
    )

    batch = next(iter(loader))
    assert batch["input_ids"].shape == (1, 4)
    assert batch["labels"].shape == (1, 4)
    assert torch.all(batch["attention_mask"] == 1)


def test_streaming_jsonl_dataloader_filters_garbage(tmp_path):
    data_path = tmp_path / "stream.jsonl"
    data_path.write_text(
        '{"prompt":"Explain gravity","completion":"Gravity attracts objects with mass."}\n'
        '{"prompt":"!!!!","completion":"@@@@####"}\n',
        encoding="utf-8",
    )

    loader = create_dataloader_from_paths(
        data_paths=(str(data_path),),
        tokenizer=FakeTokenizer(),
        batch_size=1,
        max_length=8,
        stride=4,
        num_workers=0,
        use_streaming=True,
        use_quality_filter=True,
    )

    batch = next(iter(loader))
    assert batch["input_ids"].shape == (1, 8)
    assert (batch["attention_mask"].sum().item()) > 0


def test_parse_kafka_training_message_accepts_supported_shapes():
    payload = b'{"prompt":"Explain gravity","completion":"Objects attract each other."}'
    alt_payload = '{"question":"What is 2+2?","final_answer":"4"}'

    assert parse_kafka_training_message(payload) == {
        "prompt": "Explain gravity",
        "completion": "Objects attract each other.",
    }
    assert parse_kafka_training_message(alt_payload) == {
        "prompt": "What is 2+2?",
        "completion": "4",
    }
    assert parse_kafka_training_message("not-json") is None


class FakeKafkaConsumer:
    def __init__(self, messages):
        self.messages = list(messages)
        self.closed = False
        self.validated = False

    def validate(self):
        self.validated = True

    def poll(self, timeout_s: float):
        if self.messages:
            return self.messages.pop(0)
        return None

    def close(self):
        self.closed = True


def test_kafka_dataloader_yields_training_batches():
    consumer = FakeKafkaConsumer(
        [
            b'{"prompt":"Explain gravity","completion":"Objects with mass attract each other."}',
            b'{"prompt":"!!!!","completion":"@@@@####"}',
        ]
    )

    loader = create_dataloader_from_paths(
        data_paths=(),
        tokenizer=FakeTokenizer(),
        batch_size=1,
        max_length=8,
        stride=4,
        num_workers=0,
        use_kafka=True,
        kafka_config=KafkaConfig(
            backend="confluent-kafka",
            bootstrap_servers="localhost:9092",
            topic="train-topic",
            group_id="test-group",
            max_empty_polls=1,
        ),
        kafka_consumer_factory=lambda cfg: consumer,
        use_quality_filter=True,
    )

    batch = next(iter(loader))
    assert batch["input_ids"].shape == (1, 8)
    assert batch["labels"].shape == (1, 8)
    assert batch["attention_mask"].sum().item() > 0
    assert consumer.closed is True


def test_validate_kafka_startup_uses_consumer_validation():
    consumer = FakeKafkaConsumer([])
    config = KafkaConfig(
        backend="confluent-kafka",
        bootstrap_servers="localhost:9092",
        topic="train-topic",
        group_id="test-group",
        max_empty_polls=1,
    )

    validate_kafka_startup(config, consumer_factory=lambda cfg: consumer)

    assert consumer.validated is True
    assert consumer.closed is True


def test_train_config_validates_kafka_requirements(tmp_path):
    cfg = TrainConfig(
        checkpoint_dir=str(tmp_path),
        data_path="unused.txt",
        data_paths=("unused.txt",),
        use_kafka_dataset=True,
        kafka_topic="train-topic",
        kafka_bootstrap_servers="localhost:9092",
        max_steps=100,
    )

    assert cfg.use_kafka_dataset is True
    assert cfg.kafka_topic == "train-topic"
