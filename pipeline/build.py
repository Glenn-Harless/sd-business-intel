"""Orchestrator: ingest → transform → export."""

from __future__ import annotations

import sys
import time

from pipeline.ingest_btax import ingest as ingest_btax
from pipeline.ingest_census import ingest as ingest_census
from pipeline.ingest_civic import ingest as ingest_civic
from pipeline.transform import transform


def main() -> None:
    force = "--force" in sys.argv
    t0 = time.time()

    print("=" * 60)
    print("SD Business Intel Pipeline")
    print("=" * 60)

    print("\n── Step 1: Ingest business tax certs ──")
    btax_paths = ingest_btax(force=force)
    print(f"  {len(btax_paths)} files ready\n")

    print("── Step 2: Ingest census demographics ──")
    ingest_census(force=force)
    print()

    print("── Step 3: Copy civic cross-references ──")
    civic_paths = ingest_civic(force=force)
    print(f"  {len(civic_paths)} civic files ready\n")

    print("── Step 4: Transform ──")
    transform()

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
