import hashlib
import numpy as np

def sha256(n: int, size: int = 16, output_raw=False, input_raw=False) -> int:
    if not input_raw:
        n = int(n).to_bytes(size, 'big', signed=True)
    x = hashlib.sha256(n).digest()
    if output_raw: return x
    return int.from_bytes(x, 'big', signed=True)

def pseudo(x: int, y: int, seed: int, size: int = 16):
    x = sha256(x, output_raw=True)
    y = sha256(y, output_raw=True)
    seed = sha256(seed, output_raw=True)
    total = sha256(x + y + seed, input_raw=True)
    mask = (2 ** (size + 1)) - 1
    return (total & mask) / mask


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi