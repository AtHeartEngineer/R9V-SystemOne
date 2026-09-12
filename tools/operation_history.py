"""Conservative host operation pairing; gaps never identify a faulting GPU call."""
import itertools


def analyze(records):
    identities = {
        (r["container"], r["rank"], r["pid"], r["start_ticks"]) for r in records
    }
    if len(identities) > 1:
        raise ValueError("operation records have mixed worker identities")
    sequences = [r["sequence"] for r in records]
    unique = set(sequences)
    pairs = []
    for before, after in itertools.pairwise(records):
        if (
            before["stage"] == "execute_model_enter"
            and after["stage"] == "execute_model_return"
            and after["sequence"] == before["sequence"] + 1
            and before["steps"] == after["steps"]
            and before["thread_id"] == after["thread_id"]
            and after["monotonic_ns"] >= before["monotonic_ns"]
        ):
            pairs.append(
                {
                    "enter": before,
                    "return": after,
                    "host_ms": (after["monotonic_ns"] - before["monotonic_ns"]) / 1e6,
                }
            )
    return {
        "records": len(records),
        "missing": max(unique) - min(unique) + 1 - len(unique) if unique else 0,
        "duplicates": len(sequences) - len(unique),
        "pairs": pairs,
    }
