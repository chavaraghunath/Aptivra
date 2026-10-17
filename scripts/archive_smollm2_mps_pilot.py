"""Archive completed lane-8a evidence into a compact index and quiet-record shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from aptivra.eval.pilot_evidence_archive import archive_pilot_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = archive_pilot_evidence(args.source, args.destination)
    print(
        f"Archived {result['archive']['quiet_record_count']} quiet records to "
        f"{args.destination.parent / 'quiet-records'}"
    )


if __name__ == "__main__":
    main()
