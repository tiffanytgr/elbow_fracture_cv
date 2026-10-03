"""Build a paired AP+LAT, 5-class manifest for the end-to-end baseline.

The cascade is trained from per-view manifests (``train.csv`` / ``LAT_train.csv``,
each ``Filepath,Label``). The end-to-end baseline needs one row per *patient*
with both views, so this module:

  1. walks the KKH dataset tree (or reads the existing per-view CSVs),
  2. extracts a case ID from each filename,
  3. pairs AP with LAT on that case ID,
  4. assigns the split from the existing per-view manifests so the baseline is
     evaluated on exactly the same patients as the cascade.

Step 4 is the important one for the comparison the reviewer asked for: a
baseline trained on a different split is not a baseline. If the existing
manifests are passed via ``--split-csv``, every paired case inherits its split
from them and cases whose two views disagree on split are dropped with a
warning rather than leaked.

Usage
-----
From the dataset tree, reusing the cascade's splits::

    python -m research.multiclass_baseline.manifest \
        --data-root "/path/to/Supracondylar Fracture Data Set" \
        --split-csv train=experiments/train.csv \
        --split-csv val=experiments/val.csv \
        --split-csv test=experiments/test.csv \
        --out research/outputs/paired_manifest.csv

From per-view manifests only (no tree walk)::

    python -m research.multiclass_baseline.manifest \
        --ap-csv train=experiments/train.csv --lat-csv train=experiments/LAT_train.csv \
        --ap-csv val=experiments/val.csv     --lat-csv val=experiments/LAT_val.csv \
        --ap-csv test=experiments/test.csv   --lat-csv test=experiments/LAT_test.csv \
        --out research/outputs/paired_manifest.csv

Output columns: ``case_id,split,grade,label_idx,ap_path,lat_path``.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

from .labels import GRADES, UnknownGradeError, grade_to_index, normalise_grade

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ("case_id", "split", "grade", "label_idx", "ap_path", "lat_path")

# Default case-ID pattern, derived from the filename grammar actually present
# in the KKH manifests:
#
#   A3517.png       A326 AP.png      A162 AP2.png     -> A3517, A326, A162
#   A241A AP.png    A594B AP.png     A2479A.png       -> A241A, A594B, A2479A
#   A118 (L).png    A118 (R).png                      -> A118 (L), A118 (R)
#
# The trailing letter group matters: A241 and A241A are different patients and
# carry different Gartland grades, so a pattern that stopped at the digits
# would silently merge them into one case. The view token is always separated
# by a space, so the letter group can never swallow "AP" or "LAT".
# Laterality in parentheses is kept for the same reason -- a left and a right
# elbow are separate studies.
DEFAULT_ID_PATTERN = r"^([A-Za-z]*\d+[A-Za-z]*(?:\s*\([LRlr]\))?)"

# View is identified from the path, not the filename, because many files are
# bare case IDs ("A3517.png") sitting inside a "SC Grade 2a AP" folder.
_AP_TOKENS = ("ap",)
_LAT_TOKENS = ("lat", "lateral")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ViewRecord:
    """One radiograph: where it is, what grade, which view, which split."""

    case_id: str
    path: str
    grade: str
    view: str  # "AP" or "LAT"
    split: str | None = None


@dataclass
class PairingReport:
    """Counts of what happened during pairing, for the run log."""

    ap_seen: int = 0
    lat_seen: int = 0
    paired: int = 0
    ap_only: list[str] = field(default_factory=list)
    lat_only: list[str] = field(default_factory=list)
    grade_conflict: list[str] = field(default_factory=list)
    split_conflict: list[str] = field(default_factory=list)
    unsplit: list[str] = field(default_factory=list)
    excluded_flexion: int = 0
    duplicate_views: list[str] = field(default_factory=list)

    def log_summary(self) -> None:
        log.info("AP radiographs seen:  %d", self.ap_seen)
        log.info("LAT radiographs seen: %d", self.lat_seen)
        log.info("paired cases:         %d", self.paired)
        if self.excluded_flexion:
            log.info("excluded (flexion):   %d", self.excluded_flexion)
        for name, cases in (
            ("AP without LAT", self.ap_only),
            ("LAT without AP", self.lat_only),
            ("grade disagreement between views", self.grade_conflict),
            ("split disagreement between views", self.split_conflict),
            ("no split assigned", self.unsplit),
            ("more than one file per view", self.duplicate_views),
        ):
            if cases:
                shown = ", ".join(sorted(cases)[:10])
                more = f" (+{len(cases) - 10} more)" if len(cases) > 10 else ""
                log.warning("dropped %d case(s) -- %s: %s%s", len(cases), name, shown, more)


def _as_posix(raw: str) -> str:
    """Normalise a manifest path that may be a Windows path from Colab/Drive."""
    text = str(raw).strip().strip('"')
    if "\\" in text and "/" not in text:
        return PureWindowsPath(text).as_posix()
    return text


def extract_case_id(path: str, pattern: str = DEFAULT_ID_PATTERN) -> str | None:
    """Pull the patient/case ID out of a radiograph filename."""
    stem = Path(_as_posix(path)).stem
    match = re.match(pattern, stem.strip())
    if not match:
        return None
    # Drop the space before a laterality suffix, so "A118 (L)" and
    # "A118(L)" are the same case.
    return re.sub(r"\s*\(", "(", match.group(1)).strip().upper()


def detect_view(path: str) -> str | None:
    """Infer AP vs LAT from the path components (folder name or filename).

    Checked most-specific-first: LAT before AP, because "SC Grade 2a LAT"
    contains no "ap" token but a filename like "A326 AP.png" does.
    """
    parts = [p.casefold() for p in Path(_as_posix(path)).parts]
    tokens = {t for part in parts for t in re.split(r"[^a-z0-9]+", part) if t}
    if any(t in tokens for t in _LAT_TOKENS):
        return "LAT"
    if any(t in tokens for t in _AP_TOKENS):
        return "AP"
    return None


def _read_view_csv(
    csv_path: Path,
    split: str | None,
    id_pattern: str,
    force_view: str | None,
    report: PairingReport,
) -> list[ViewRecord]:
    """Read one ``Filepath,Label`` manifest into ViewRecords."""
    records: list[ViewRecord] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty")
        cols = {c.strip().casefold(): c for c in reader.fieldnames}
        try:
            path_col, label_col = cols["filepath"], cols["label"]
        except KeyError as exc:
            raise ValueError(
                f"{csv_path} must have 'Filepath' and 'Label' columns, found "
                f"{reader.fieldnames}"
            ) from exc

        for line_no, row in enumerate(reader, start=2):
            raw_path = row[path_col]
            if not raw_path or not raw_path.strip():
                continue
            try:
                grade = normalise_grade(row[label_col])
            except UnknownGradeError as exc:
                raise ValueError(f"{csv_path}:{line_no}: {exc}") from exc
            if grade is None:
                report.excluded_flexion += 1
                continue
            case_id = extract_case_id(raw_path, id_pattern)
            if case_id is None:
                log.warning("%s:%d: no case ID in %r -- skipped", csv_path, line_no, raw_path)
                continue
            view = force_view or detect_view(raw_path)
            if view is None:
                log.warning("%s:%d: cannot tell AP from LAT in %r -- skipped",
                            csv_path, line_no, raw_path)
                continue
            records.append(
                ViewRecord(case_id=case_id, path=_as_posix(raw_path),
                           grade=grade, view=view, split=split)
            )
    return records


def scan_data_root(
    data_root: Path, id_pattern: str, report: PairingReport
) -> list[ViewRecord]:
    """Walk the KKH dataset tree, taking the grade from the folder name.

    Expects the shipped layout, e.g.
    ``<root>/SC Grade 2a/SC Grade 2a AP/A3517.png`` and
    ``<root>/Normal/Normal LAT/A118.png``. Any directory whose name maps onto
    the canonical vocabulary is used as the grade for files beneath it.
    """
    records: list[ViewRecord] = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(data_root)
        grade = None
        for part in rel.parts[:-1]:
            try:
                candidate = normalise_grade(part)
            except UnknownGradeError:
                continue
            if candidate is None:  # flexion folder
                grade = "__excluded__"
                break
            grade = candidate
        if grade == "__excluded__":
            report.excluded_flexion += 1
            continue
        if grade is None:
            continue
        case_id = extract_case_id(path.name, id_pattern)
        view = detect_view(str(rel))
        if case_id is None or view is None:
            log.debug("skipped %s (case_id=%s view=%s)", rel, case_id, view)
            continue
        records.append(
            ViewRecord(case_id=case_id, path=path.as_posix(), grade=grade, view=view)
        )
    return records


def apply_splits(
    records: list[ViewRecord],
    split_records: list[ViewRecord],
) -> None:
    """Stamp each record's split from the per-view split manifests.

    Matching is on (case_id, view) first and falls back to case_id, so a case
    listed in the AP manifest also places its LAT view in the same split. This
    is what keeps the baseline's test set identical to the cascade's.
    """
    by_case_view: dict[tuple[str, str], str] = {}
    by_case: dict[str, set[str]] = defaultdict(set)
    for rec in split_records:
        if rec.split is None:
            continue
        by_case_view[(rec.case_id, rec.view)] = rec.split
        by_case[rec.case_id].add(rec.split)

    for rec in records:
        split = by_case_view.get((rec.case_id, rec.view))
        if split is None:
            candidates = by_case.get(rec.case_id, set())
            split = next(iter(candidates)) if len(candidates) == 1 else None
        rec.split = split


def pair_records(
    records: list[ViewRecord], report: PairingReport
) -> list[dict[str, object]]:
    """Collapse per-view records into one paired row per case."""
    grouped: dict[str, dict[str, list[ViewRecord]]] = defaultdict(
        lambda: {"AP": [], "LAT": []}
    )
    for rec in records:
        grouped[rec.case_id][rec.view].append(rec)
        if rec.view == "AP":
            report.ap_seen += 1
        else:
            report.lat_seen += 1

    rows: list[dict[str, object]] = []
    for case_id in sorted(grouped):
        views = grouped[case_id]
        ap_list, lat_list = views["AP"], views["LAT"]
        if not ap_list:
            report.lat_only.append(case_id)
            continue
        if not lat_list:
            report.ap_only.append(case_id)
            continue
        if len(ap_list) > 1 or len(lat_list) > 1:
            # Deterministic choice: first by sorted path, and record it so the
            # count appears in the run log rather than passing silently.
            report.duplicate_views.append(case_id)
            ap_list = sorted(ap_list, key=lambda r: r.path)
            lat_list = sorted(lat_list, key=lambda r: r.path)
        ap, lat = ap_list[0], lat_list[0]

        if ap.grade != lat.grade:
            report.grade_conflict.append(case_id)
            continue
        splits = {r.split for r in (ap, lat) if r.split is not None}
        if len(splits) > 1:
            report.split_conflict.append(case_id)
            continue
        if not splits:
            report.unsplit.append(case_id)
            continue

        label_idx = grade_to_index(ap.grade)
        assert label_idx is not None  # flexion already filtered out
        rows.append(
            {
                "case_id": case_id,
                "split": splits.pop(),
                "grade": ap.grade,
                "label_idx": label_idx,
                "ap_path": ap.path,
                "lat_path": lat.path,
            }
        )
        report.paired += 1
    return rows


def write_manifest(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def log_distribution(rows: list[dict[str, object]]) -> None:
    """Print the split x grade table so it can be checked against Table 1."""
    counts: Counter[tuple[str, str]] = Counter(
        (str(r["split"]), str(r["grade"])) for r in rows
    )
    splits = sorted({s for s, _ in counts})
    header = f"{'grade':<10}" + "".join(f"{s:>8}" for s in splits) + f"{'total':>8}"
    log.info("paired-case distribution")
    log.info(header)
    for grade in GRADES:
        cells = [counts.get((s, grade), 0) for s in splits]
        log.info(f"{grade:<10}" + "".join(f"{c:>8}" for c in cells) + f"{sum(cells):>8}")
    totals = [sum(counts.get((s, g), 0) for g in GRADES) for s in splits]
    log.info(f"{'total':<10}" + "".join(f"{t:>8}" for t in totals) + f"{sum(totals):>8}")


def _parse_keyed(values: list[str] | None, flag: str) -> dict[str, Path]:
    """Parse repeated ``--flag split=path`` arguments."""
    out: dict[str, Path] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"{flag} expects split=path, got {item!r}")
        split, _, path = item.partition("=")
        split = split.strip()
        if not split:
            raise SystemExit(f"{flag} expects a non-empty split name, got {item!r}")
        out[split] = Path(path.strip()).expanduser()
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a paired AP+LAT 5-class manifest for the baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-root", type=Path,
                        help="Root of the KKH dataset tree; grade is taken from folder names.")
    parser.add_argument("--ap-csv", action="append", metavar="SPLIT=PATH",
                        help="Per-view AP manifest for a split (repeatable).")
    parser.add_argument("--lat-csv", action="append", metavar="SPLIT=PATH",
                        help="Per-view LAT manifest for a split (repeatable).")
    parser.add_argument("--split-csv", action="append", metavar="SPLIT=PATH",
                        help="Manifest used only to assign splits when scanning "
                             "--data-root (repeatable).")
    parser.add_argument("--out", type=Path, required=True, help="Output manifest CSV.")
    parser.add_argument("--id-pattern", default=DEFAULT_ID_PATTERN,
                        help=f"Case-ID regex, group 1 is the ID (default: {DEFAULT_ID_PATTERN!r}).")
    parser.add_argument("--allow-unpaired", action="store_true",
                        help="Exit 0 even if no case could be paired (for dry runs).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    report = PairingReport()
    records: list[ViewRecord] = []

    ap_csvs = _parse_keyed(args.ap_csv, "--ap-csv")
    lat_csvs = _parse_keyed(args.lat_csv, "--lat-csv")
    split_csvs = _parse_keyed(args.split_csv, "--split-csv")

    if args.data_root:
        if not args.data_root.is_dir():
            raise SystemExit(f"--data-root {args.data_root} is not a directory")
        log.info("scanning %s", args.data_root)
        records.extend(scan_data_root(args.data_root, args.id_pattern, report))
        split_records: list[ViewRecord] = []
        for split, path in split_csvs.items():
            split_records.extend(
                _read_view_csv(path, split, args.id_pattern, None, PairingReport())
            )
        if split_records:
            apply_splits(records, split_records)
        elif not split_csvs:
            log.warning("no --split-csv given; every case will be dropped as unsplit")
    else:
        if not ap_csvs or not lat_csvs:
            raise SystemExit(
                "give --data-root, or both --ap-csv and --lat-csv for each split"
            )
        for split, path in ap_csvs.items():
            records.extend(_read_view_csv(path, split, args.id_pattern, "AP", report))
        for split, path in lat_csvs.items():
            records.extend(_read_view_csv(path, split, args.id_pattern, "LAT", report))

    rows = pair_records(records, report)
    report.log_summary()
    if not rows:
        log.error("no paired cases produced -- check --id-pattern and the split CSVs")
        if not args.allow_unpaired:
            return 1
    else:
        log_distribution(rows)
    write_manifest(rows, args.out)
    log.info("wrote %d paired cases to %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
