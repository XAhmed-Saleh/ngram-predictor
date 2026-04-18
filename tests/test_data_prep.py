"""Unit tests for the Normalizer class (src/data_prep/normalizer.py)."""

import os
import tempfile
import pytest
from src.data_prep.normalizer import Normalizer


@pytest.fixture
def normalizer():
    """Provide a fresh Normalizer instance."""
    return Normalizer()


# --- normalize() tests: each step independently ---

class TestNormalizeLowercase:
    def test_normalize_lowercases_text(self, normalizer):
        result = normalizer.normalize("HELLO WORLD")
        assert result == "hello world"

    def test_normalize_lowercases_mixed_case(self, normalizer):
        result = normalizer.normalize("HeLLo WoRLd")
        assert result == "hello world"


class TestNormalizePunctuation:
    def test_normalize_removes_punctuation(self, normalizer):
        result = normalizer.normalize("hello, world!")
        assert result == "hello world"

    def test_normalize_removes_various_punctuation(self, normalizer):
        result = normalizer.normalize("it's a test—with (special) chars: yes.")
        assert "," not in result
        assert "." not in result
        assert "(" not in result
        assert ")" not in result


class TestNormalizeNumbers:
    def test_normalize_removes_numbers(self, normalizer):
        result = normalizer.normalize("chapter 3 section 12")
        assert "3" not in result
        assert "12" not in result
        assert "chapter" in result
        assert "section" in result


class TestNormalizeWhitespace:
    def test_normalize_strips_extra_whitespace(self, normalizer):
        result = normalizer.normalize("  hello   world  ")
        assert result == "hello world"

    def test_normalize_strips_newlines(self, normalizer):
        result = normalizer.normalize("hello\n\n\nworld")
        assert result == "hello world"


class TestNormalizeSequence:
    def test_normalize_applies_all_steps_in_order(self, normalizer):
        result = normalizer.normalize("  HELLO, World! Chapter 7.  ")
        assert result == "hello world chapter"

    def test_normalize_empty_string(self, normalizer):
        result = normalizer.normalize("")
        assert result == ""


# --- strip_gutenberg() ---

class TestStripGutenberg:
    def test_strip_gutenberg_removes_header_and_footer(self, normalizer):
        text = (
            "Some preamble text\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "Actual content here.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "Some footer text\n"
        )
        result = normalizer.strip_gutenberg(text)
        assert "preamble" not in result
        assert "footer" not in result
        assert "Actual content here." in result

    def test_strip_gutenberg_no_markers(self, normalizer):
        text = "Just plain text without markers."
        result = normalizer.strip_gutenberg(text)
        assert result == text


# --- sentence_tokenize() ---

class TestSentenceTokenize:
    def test_returns_list_with_at_least_one_element(self, normalizer):
        result = normalizer.sentence_tokenize("hello world this is a test")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_splits_multiple_sentences(self, normalizer):
        result = normalizer.sentence_tokenize("first sentence. second sentence. third one.")
        assert len(result) >= 2


# --- word_tokenize() ---

class TestWordTokenize:
    def test_returns_list_of_strings(self, normalizer):
        result = normalizer.word_tokenize("hello world")
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_no_empty_tokens(self, normalizer):
        result = normalizer.word_tokenize("hello   world  test")
        assert "" not in result
        assert all(len(t) > 0 for t in result)

    def test_splits_on_whitespace(self, normalizer):
        result = normalizer.word_tokenize("the cat sat")
        assert result == ["the", "cat", "sat"]


# --- load() ---

class TestLoad:
    def test_load_reads_txt_files(self, normalizer, tmp_path):
        (tmp_path / "book1.txt").write_text("Hello world.", encoding="utf-8")
        (tmp_path / "book2.txt").write_text("Goodbye world.", encoding="utf-8")
        result = normalizer.load(str(tmp_path))
        assert "Hello world." in result
        assert "Goodbye world." in result

    def test_load_raises_on_missing_folder(self, normalizer):
        with pytest.raises(FileNotFoundError):
            normalizer.load("/nonexistent/folder/path")


# --- save() ---

class TestSave:
    def test_save_writes_sentences(self, normalizer, tmp_path):
        filepath = str(tmp_path / "output.txt")
        sentences = [["the", "cat", "sat"], ["on", "the", "mat"]]
        normalizer.save(sentences, filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert lines[0].strip() == "the cat sat"
        assert lines[1].strip() == "on the mat"
