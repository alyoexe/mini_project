import argparse
import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)


def split_feedback(records):
    yes_cases = []
    no_cases = []
    skipped = []

    for row in records:
        decision = row.get("decision", "").lower()
        if decision == "yes":
            yes_cases.append(row)
        elif decision == "no":
            no_cases.append(row)
        else:
            skipped.append(row)

    return yes_cases, no_cases, skipped


def build_summary(yes_cases, no_cases, skipped):
    class_breakdown = {}
    for row in yes_cases + no_cases:
        name = row.get("class_name", "unknown")
        class_breakdown[name] = class_breakdown.get(name, 0) + 1

    return {
        "total_records": len(yes_cases) + len(no_cases) + len(skipped),
        "yes_records": len(yes_cases),
        "no_records": len(no_cases),
        "skipped_records": len(skipped),
        "class_breakdown": class_breakdown,
        "notes": [
            "yes_records: model prediction accepted by human",
            "no_records: model prediction rejected by human and should be manually corrected before retraining",
            "skipped_records: prompt timed out or no decision",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare HITL feedback manifests for periodic retraining.")
    parser.add_argument("--feedback-log", default="feedback_data/feedback_log.jsonl", help="Path to JSONL feedback log")
    parser.add_argument("--out-dir", default="feedback_data/prepared", help="Output directory for manifests")
    args = parser.parse_args()

    log_path = Path(args.feedback_log)
    out_dir = Path(args.out_dir)

    records = read_jsonl(log_path)
    yes_cases, no_cases, skipped = split_feedback(records)

    write_json(out_dir / "yes_manifest.json", yes_cases)
    write_json(out_dir / "no_review_manifest.json", no_cases)
    write_json(out_dir / "skipped_manifest.json", skipped)

    summary = build_summary(yes_cases, no_cases, skipped)
    write_json(out_dir / "summary.json", summary)

    print("Prepared feedback manifests:")
    print(f"  yes: {len(yes_cases)}")
    print(f"  no: {len(no_cases)}")
    print(f"  skipped: {len(skipped)}")
    print(f"  summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
