from __future__ import annotations

import argparse
import logging
import timeit

import yaml

from jax_benchmark.io import get_machine_info, mute_warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        "-o",
        help="YAML output filename for the benchmark results",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--shape",
        "-s",
        default="5000x100",
        help="Shape of the array to take a dot product of",
        type=str,
    )
    parser.add_argument(
        "--repeat",
        "-r",
        default=10,
        help="Number of repeats",
        type=int,
    )
    parser.add_argument(
        "--number",
        "-n",
        default=5,
        help="Number of times that the job should be run on a single repeat",
        type=int,
    )
    args = parser.parse_args()
    mute_warnings()
    benchmarks = run_benchmark(
        shape=tuple(  # pyright:ignore[reportArgumentType]
            int(i) for i in args.shape.split("x")
        ),
        repeat=args.repeat,
        number=args.number,
    )
    output_file = args.output
    with open(output_file, "w") as f:
        yaml.safe_dump(benchmarks, f, sort_keys=False)
    logging.info(f"Benchmark written to {output_file}")
    return 0


def run_benchmark(shape: tuple[int, int], number: int, repeat: int) -> dict:
    import jax.numpy as jnp
    from jax import random

    array = random.normal(
        key=random.PRNGKey(0),
        shape=shape,
    )

    def run() -> None:
        jnp.dot(array, array.T).block_until_ready()

    return {
        "results_in_seconds": timeit.repeat(run, number=number, repeat=repeat),
        "number_of_runs": number,
        "repeat": repeat,
        "shape": "x".join(map(str, shape)),
        "machine_info": get_machine_info(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
