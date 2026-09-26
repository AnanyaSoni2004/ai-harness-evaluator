from toolkit.text import slugify


def test_lowercases():
    assert slugify("Hello") == "hello"


def test_keeps_letters_and_digits():
    assert slugify("abc123") == "abc123"


def test_single_space_becomes_hyphen():
    assert slugify("hello world") == "hello-world"
