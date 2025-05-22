from collections import defaultdict
import functools
import time
from typing import Any, Callable, Dict, Tuple


_profile_data: Dict[str, Tuple[int, float]] = defaultdict(lambda: [0, 0.0])

def time_avg(fn: Callable) -> Callable:
    """Декоратор для измерения среднего wall-clock-времени вызова."""
    qual = fn.__qualname__                    # Имя включая класс (для методов)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            stat = _profile_data[qual]
            stat[0] += 1          # calls
            stat[1] += dt         # total time

    return wrapper

def get_profile(min_calls: int = 1, top: int | None = None) -> None:
    """
    Выводит таблицу <среднее время> для всех функций,
    у которых вызовов ≥ min_calls.
    """
    rows = [
        (name, calls, total / calls * 1_000)   # ms
        for name, (calls, total) in _profile_data.items()
        if calls >= min_calls
    ]
    rows.sort(key=lambda r: r[2], reverse=True)   # медленные сверху
    if top:
        rows = rows[:top]

    _profile_data.clear()
    return rows