"""
data_source/tokenizer.py
------------------------
Production-grade tokenizer wrapper for decoder-only LLM training.

Design Goals
------------
- Stable vocabulary handling (no silent resizing issues)
- Decoder-only aligned
- Strict validation
- Safe special token registration
- Conversation formatting helpers
- Encode/decode behavior
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import AddedToken


class Tokenizer:
    """
    FTK wrapper `tokenizers` JSON tokenizer.

    This class is designed for decoder-only LLMs (FTK style).

    Parameters
    ----------
    tokenizer_path : str
        Path to tokenizer.json
    add_conversation_tokens : bool
        Whether to register conversation markers if missing.
    """

    # Default decoder-style special tokens
    DEFAULT_SPECIAL_TOKENS = {
        "pad": "[PAD]",
        "unk": "[UNK]",
        "bos": "<|bos|>",
        "eos": "<|eot|>",
    }

    # Conversation tokens
    CONVERSATION_TOKENS = {
        "system": "<|system|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
        "eot": "<|eot|>",
    }

    def __init__(
        self,
        tokenizer_path: str = "tokenizer.json",
        add_conversation_tokens: bool = True,
    ) -> None:
        path = Path(tokenizer_path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        self.tokenizer_path = str(path)
        self._tok: HFTokenizer = HFTokenizer.from_file(str(path))

        # Declare dynamic attributes to satisfy mypy
        self.pad_token: str
        self.pad_token_id: int
        self.unk_token: str
        self.unk_token_id: int
        self.bos_token: str
        self.bos_token_id: int
        self.eos_token: str
        self.eos_token_id: int
        self.system_token: str
        self.system_token_id: int
        self.user_token: str
        self.user_token_id: int
        self.assistant_token: str
        self.assistant_token_id: int
        self.eot_token: str
        self.eot_token_id: int
        self.turn_end_token: str
        self.turn_end_token_id: int

        # Register base special tokens (must exist in vocab)
        self._init_base_special_tokens()

        # Optionally register conversation markers safely
        if add_conversation_tokens:
            self._register_conversation_tokens()

        # Cache vocab size after all additions
        self._vocab_size = self._tok.get_vocab_size()

        # Final sanity validation
        self._validate_special_tokens()

    # ------------------------------------------------------------------
    # Internal Initialization
    # ------------------------------------------------------------------

    def _init_base_special_tokens(self) -> None:
        """
        Ensure required decoder-only tokens exist.
        """
        for name, token in self.DEFAULT_SPECIAL_TOKENS.items():
            token_id = self._tok.token_to_id(token)
            if token_id is None:
                raise ValueError(
                    f"Required special token '{token}' missing from tokenizer.json.\n"
                    f"Make sure your tokenizer was trained with decoder-style tokens."
                )
            setattr(self, f"{name}_token", token)
            setattr(self, f"{name}_token_id", token_id)

    def _register_conversation_tokens(self) -> None:
        """
        Register conversation tokens only if missing.
        Safe: does not duplicate entries.
        """
        new_tokens = []

        for name, token in self.CONVERSATION_TOKENS.items():
            if self._tok.token_to_id(token) is None:
                new_tokens.append(
                    AddedToken(
                        token,
                        single_word=False,
                        lstrip=False,
                        rstrip=False,
                        special=True,
                    )
                )

        if new_tokens:
            self._tok.add_special_tokens(new_tokens)

        # Cache IDs
        for name, token in self.CONVERSATION_TOKENS.items():
            token_id = self._tok.token_to_id(token)
            if token_id is None:
                raise ValueError(f"Failed to register conversation token: {token}")
            setattr(self, f"{name}_token", token)
            setattr(self, f"{name}_token_id", token_id)

        # Alias for compatibility
        self.turn_end_token = self.eot_token
        self.turn_end_token_id = self.eot_token_id

    def _validate_special_tokens(self) -> None:
        """
        Ensure all required tokens resolve correctly.
        """
        required = [
            "pad_token_id",
            "unk_token_id",
            "bos_token_id",
            "eos_token_id",
        ]
        for attr in required:
            if getattr(self, attr, None) is None:
                raise RuntimeError(f"Tokenizer missing required attribute: {attr}")

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode text into token IDs.

        Parameters
        ----------
        text : str
        add_bos : bool
        add_eos : bool
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")

        ids = self._tok.encode(text, add_special_tokens=False).ids

        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = True,
        clean_up: bool = True,
    ) -> str:
        """
        Decode token IDs into text.
        """
        token_ids = list(token_ids)

        if not all(isinstance(i, int) for i in token_ids):
            raise TypeError("All token IDs must be integers.")

        text = self._tok.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

        if clean_up:
            text = self._clean_text(text)

        return text

    # ------------------------------------------------------------------
    # Conversation Formatting
    # ------------------------------------------------------------------

    def format_chat(
        self,
        system: Optional[str],
        user: str,
        assistant: Optional[str] = None,
    ) -> List[int]:
        """
        Format a conversation turn into tokens.

        Structure:
        <|system|> ... <|eot|>
        <|user|> ... <|eot|>
        <|assistant|> ... <|eot|>
        """
        tokens: List[int] = []

        if system:
            tokens += [self.system_token_id]
            tokens += self.encode(system)
            tokens += [self.eot_token_id]

        tokens += [self.user_token_id]
        tokens += self.encode(user)
        tokens += [self.eot_token_id]

        if assistant:
            tokens += [self.assistant_token_id]
            tokens += self.encode(assistant)
            tokens += [self.eot_token_id]

        return tokens

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Safe whitespace cleanup for display only.
        """
        if "Ġ" in text or "Ċ" in text:
            text = text.replace("Ġ", " ").replace("Ċ", "\n")

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" \n", "\n", text)
        text = re.sub(r"\n ", "\n", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)

        return text.strip()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        return self._tok.id_to_token(token_id)

    def __repr__(self) -> str:
        return (
            f"Tokenizer(path='{self.tokenizer_path}', "
            f"vocab_size={self.vocab_size})"
        )