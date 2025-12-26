from src.utils.text_preprocessing import (
    tokenize, 
    text_to_indices, 
    dedup_preprocess, 
    ngram_preprocess,
    default_preprocess,
    pad_sequence,
    build_vocab,
)


def test_tokenize():
    text = "Hello world!"
    assert tokenize(text) == ["Hello", "world!"]

    text = "This is a test"
    assert tokenize(text) == ["This", "is", "a", "test"]


def test_default_preprocess():
    text = "  Hello World!  "
    assert default_preprocess(text) == ["hello", "world!"]

    text = "TESTING preprocess"
    assert default_preprocess(text) == ["testing", "preprocess"]
    
def test_ngram_preprocess():
    text = "I love programming"
    
    # Test for bigrams (n=2)
    assert ngram_preprocess(text, 2) == ["i_love", "love_programming"]
    
    # Test for trigrams (n=3)
    assert ngram_preprocess(text, 3) == ["i_love_programming"]

    # Test for unigrams (n=1)
    assert ngram_preprocess(text, 1) ==  ["i", "love", "programming"]
    
    
def test_dedup_preprocess():
    text = "Open Read Read Read Close Open"
    assert dedup_preprocess(text) == ["open", "read", "close", "open"]

    text = "Hello Hello Hello"
    assert dedup_preprocess(text) == ["hello"]
    

def test_text_to_indices():
    texts = ["I love programming", "I love coding", "I love AI"]
    vocab = build_vocab(texts, min_freq=1)
    
    text = "I love AI"
    expected = [vocab.get("i", 0), vocab.get("love", 0), vocab.get("ai", 0)]
    assert text_to_indices(text, vocab) == expected
    
    
def test_pad_sequence():
    seq = [1, 2, 3]
    max_len = 5
    assert pad_sequence(seq, max_len) == [1, 2, 3, 1, 1]

    seq = [1, 2, 3, 4, 5, 6]
    max_len = 4
    assert pad_sequence(seq, max_len) ==  [1, 2, 3, 4]
    


# 组合测试
def test_full_preprocessing():
    text = "Open Read Read Read Close Open"
    preprocessed = dedup_preprocess(text)
    print(preprocessed)
    
    ngrammed = ngram_preprocess(" ".join(preprocessed), 2)
    assert ngrammed == ['open_read', 'read_close', 'close_open']
    print(ngrammed)
    
    vocab = build_vocab(texts=ngrammed, min_freq=1)
    print(vocab)
    
    indices = text_to_indices(" ".join(ngrammed), vocab)
    print(indices)
    
    assert indices == [vocab.get("open_read", 0), vocab.get("read_close"), vocab.get("close_open")]