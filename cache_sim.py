#!/usr/bin/env python3

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


OUTPUT_L1_HIT = "L1HIT"
OUTPUT_L2_HIT = "L2HIT"
OUTPUT_MEM_ACCESS = "MEMACC"
ADDRESS_BITS = 32
MAX_ADDRESS = 1 << ADDRESS_BITS


class CacheSimError(ValueError):
    """Raised for invalid input files or invalid simulator parameters."""


@dataclass(frozen=True)
class CacheConfig:
    line_size: int
    inclusive: bool
    l1_num_ways: int
    l1_data_size: int
    l2_num_ways: int
    l2_data_size: int

    @staticmethod
    def from_file(path: str | Path) -> "CacheConfig":
        text = Path(path).read_text(encoding="utf-8").strip()
        if not text:
            raise CacheSimError("configuration file is empty")

        config_line = ",".join(line.strip() for line in text.splitlines() if line.strip())
        parts = [part.strip() for part in config_line.split(",")]
        if len(parts) != 6:
            raise CacheSimError(
                "configuration must contain exactly 6 comma-separated fields"
            )

        try:
            line_size = int(parts[0], 10)
            inclusive_text = parts[1].upper()
            l1_num_ways = int(parts[2], 10)
            l1_data_size = int(parts[3], 10)
            l2_num_ways = int(parts[4], 10)
            l2_data_size = int(parts[5], 10)
        except ValueError as exc:
            raise CacheSimError("numeric configuration fields must be integers") from exc

        if inclusive_text not in {"TRUE", "FALSE"}:
            raise CacheSimError("CACHE_INCLUSIVE must be TRUE or FALSE")

        cfg = CacheConfig(
            line_size=line_size,
            inclusive=(inclusive_text == "TRUE"),
            l1_num_ways=l1_num_ways,
            l1_data_size=l1_data_size,
            l2_num_ways=l2_num_ways,
            l2_data_size=l2_data_size,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        numeric_values = (
            self.line_size,
            self.l1_num_ways,
            self.l1_data_size,
            self.l2_num_ways,
            self.l2_data_size,
        )
        if any(value <= 0 for value in numeric_values):
            raise CacheSimError("all numeric configuration values must be positive")
        if not all(is_power_of_two(value) for value in numeric_values):
            raise CacheSimError("all numeric configuration values must be powers of two")
        if self.line_size < 2:
            raise CacheSimError("CACHE_LINE_SIZE must be at least 2")
        if self.l1_data_size < self.line_size * self.l1_num_ways:
            raise CacheSimError("L1_DATA_SIZE is too small for the requested associativity")
        if self.l2_data_size < self.line_size * self.l2_num_ways:
            raise CacheSimError("L2_DATA_SIZE is too small for the requested associativity")
        if self.l2_data_size <= self.l1_data_size:
            raise CacheSimError("L2 must contain more data than L1")


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


class SetAssociativeCache:

    def __init__(self, *, line_size: int, num_ways: int, data_size: int) -> None:
        self.line_size = line_size
        self.num_ways = num_ways
        self.data_size = data_size
        self.num_lines = data_size // line_size
        self.num_sets = self.num_lines // num_ways
        self._sets: list[OrderedDict[int, None]] = [
            OrderedDict() for _ in range(self.num_sets)
        ]

    def _set_index_for_block(self, block_addr: int) -> int:
        return block_addr % self.num_sets

    def _set_for_block(self, block_addr: int) -> OrderedDict[int, None]:
        return self._sets[self._set_index_for_block(block_addr)]

    def contains(self, block_addr: int) -> bool:
        return block_addr in self._set_for_block(block_addr)

    def access_if_present(self, block_addr: int) -> bool:
        """Return True on hit and update LRU state."""
        cache_set = self._set_for_block(block_addr)
        if block_addr not in cache_set:
            return False
        cache_set.move_to_end(block_addr)
        return True

    def insert_or_touch(self, block_addr: int) -> int | None:

        cache_set = self._set_for_block(block_addr)
        if block_addr in cache_set:
            cache_set.move_to_end(block_addr)
            return None

        cache_set[block_addr] = None
        if len(cache_set) <= self.num_ways:
            return None

        evicted_block, _ = cache_set.popitem(last=False)
        return evicted_block

    def remove(self, block_addr: int) -> bool:
        cache_set = self._set_for_block(block_addr)
        if block_addr not in cache_set:
            return False
        del cache_set[block_addr]
        return True


class CacheHierarchySimulator:

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.l1 = SetAssociativeCache(
            line_size=config.line_size,
            num_ways=config.l1_num_ways,
            data_size=config.l1_data_size,
        )
        self.l2 = SetAssociativeCache(
            line_size=config.line_size,
            num_ways=config.l2_num_ways,
            data_size=config.l2_data_size,
        )

    def access(self, address: int, operation: str) -> str:
        block_addr = address // self.config.line_size
        operation = operation.upper()

        if operation == "R":
            return self._read(block_addr)
        if operation == "W":
            return self._write(block_addr)
        raise CacheSimError(f"unsupported memory operation: {operation!r}")

    def _read(self, block_addr: int) -> str:
        if self.l1.access_if_present(block_addr):
            return OUTPUT_L1_HIT

        if self.l2.access_if_present(block_addr):
            self._load_to_l1(block_addr)
            return OUTPUT_L2_HIT

        self._load_to_l2(block_addr)
        self._load_to_l1(block_addr)
        return OUTPUT_MEM_ACCESS

    def _write(self, block_addr: int) -> str:
        if self.l1.access_if_present(block_addr):
            self.l2.access_if_present(block_addr)
            return OUTPUT_L1_HIT

        if self.l2.access_if_present(block_addr):
            return OUTPUT_L2_HIT

        return OUTPUT_MEM_ACCESS

    def _load_to_l1(self, block_addr: int) -> None:
        self.l1.insert_or_touch(block_addr)

    def _load_to_l2(self, block_addr: int) -> None:
        evicted_from_l2 = self.l2.insert_or_touch(block_addr)
        if self.config.inclusive and evicted_from_l2 is not None:
            self.l1.remove(evicted_from_l2)


def parse_trace_line(line: str, line_number: int) -> tuple[int, str]:
    stripped = line.strip()
    if not stripped:
        raise CacheSimError("empty trace lines should be skipped before parsing")

    parts = [part.strip() for part in stripped.split(",")]
    if len(parts) != 2:
        raise CacheSimError(f"invalid trace line {line_number}: expected '<ADDRESS> , R/W'")

    address_text, operation = parts
    if not address_text:
        raise CacheSimError(f"invalid trace line {line_number}: missing address")
    if operation not in {"R", "W", "r", "w"}:
        raise CacheSimError(f"invalid trace line {line_number}: operation must be R or W")

    try:
        address = int(address_text, 16)
    except ValueError as exc:
        raise CacheSimError(f"invalid trace line {line_number}: address must be hexadecimal") from exc

    if not (0 <= address < MAX_ADDRESS):
        raise CacheSimError(f"invalid trace line {line_number}: address must fit in 32 bits")

    return address, operation.upper()


def read_trace(path: str | Path) -> Iterable[tuple[int, str]]:
    with Path(path).open("r", encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            if not line.strip():
                continue
            yield parse_trace_line(line, line_number)


def simulate(config_file: str | Path, trace_file: str | Path, output_file: str | Path) -> None:
    config = CacheConfig.from_file(config_file)
    simulator = CacheHierarchySimulator(config)

    results = [
        simulator.access(address, operation)
        for address, operation in read_trace(trace_file)
    ]

    Path(output_file).write_text("\n".join(results) + ("\n" if results else ""), encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "Usage: python cache_sim.py <config_file> <trace_file> <output_file>",
            file=sys.stderr,
        )
        return 2

    try:
        simulate(argv[1], argv[2], argv[3])
    except (OSError, CacheSimError) as exc:
        print(f"cache_sim.py: error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
