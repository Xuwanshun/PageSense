from api.auth.passwords import hash_password, verify_password


def test_hash_returns_nonempty_string():
    assert len(hash_password("mysecret")) > 0


def test_verify_correct_password():
    h = hash_password("correct")
    assert verify_password("correct", h) is True


def test_verify_wrong_password():
    h = hash_password("correct")
    assert verify_password("wrong", h) is False


def test_two_hashes_of_same_password_differ():
    assert hash_password("same") != hash_password("same")
