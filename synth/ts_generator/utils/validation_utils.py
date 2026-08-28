def validate_sequence_length(
    seq_len: int,
    min_length: int = 5
) -> None:
    if not isinstance(seq_len, int):
        raise TypeError(f"seq_len must be an integer, got {type(seq_len)}")
    if seq_len < min_length:
        raise ValueError(f"seq_len must be at least {min_length}, got {seq_len}")

def validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")

def validate_probability(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a probability in [0, 1], got {value}")
