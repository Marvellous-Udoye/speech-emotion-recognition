import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_predictions(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _safe_avg(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def _clean_value(value):
    if value is None:
        return "Unknown"
    if isinstance(value, str):
        cleaned = " ".join(value.split())
        return cleaned or "Unknown"
    return str(value)


def _normalize_profile(session):
    if not isinstance(session, dict):
        return {}
    profile = session.get("profile", {})
    if not isinstance(profile, dict):
        profile = {}

    def pick(*keys):
        for key in keys:
            if key in profile:
                return _clean_value(profile.get(key))
            if key in session:
                return _clean_value(session.get(key))
        return "Unknown"

    return {
        "gender": pick("gender"),
        "age_group": pick("age_group", "age"),
        "institution": pick("institution"),
        "level": pick("level"),
        "faculty": pick("faculty"),
        "presentation_type": pick("presentation_type", "presentation"),
        "experience": pick("experience"),
    }


def _group_key(profile, fields):
    return tuple(profile.get(field, "Unknown") for field in fields)


def aggregate_sessions(users_dir: Path):
    groups = defaultdict(list)
    totals = {
        "devices": 0,
        "recordings": 0,
        "confidences": [],
        "emotion_counts": Counter(),
    }

    for device_dir in users_dir.iterdir():
        if not device_dir.is_dir():
            continue
        session_path = device_dir / "session.json"
        preds_path = device_dir / "predictions.jsonl"
        session = _load_json(session_path) or {}
        profile = _normalize_profile(session)

        predictions = list(_iter_predictions(preds_path))
        if not predictions:
            continue

        confidences = [p.get("confidence", 0.0) for p in predictions]
        labels = [p.get("label", "unknown") for p in predictions]
        label_counts = Counter(labels)

        totals["devices"] += 1
        totals["recordings"] += len(predictions)
        totals["confidences"].extend(confidences)
        totals["emotion_counts"].update(label_counts)

        group_fields = [
            "gender",
            "age_group",
            "institution",
            "level",
            "faculty",
            "presentation_type",
            "experience",
        ]
        key = _group_key(profile, group_fields)
        groups[key].append(
            {
                "device_id": device_dir.name,
                "profile": profile,
                "recordings": len(predictions),
                "avg_confidence": _safe_avg(confidences),
                "top_emotion": label_counts.most_common(1)[0][0],
            }
        )

    return groups, totals


def write_outputs(groups, totals, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "session_summary.csv"
    json_path = out_dir / "session_summary.json"
    md_path = out_dir / "session_summary.md"
    insights_path = out_dir / "cohort_insights.md"
    insights_json_path = out_dir / "cohort_insights.json"

    rows = []
    for key, members in groups.items():
        recordings = sum(m["recordings"] for m in members)
        avg_conf = _safe_avg([m["avg_confidence"] for m in members])
        top_emotions = Counter(m["top_emotion"] for m in members)
        rows.append(
            {
                "gender": key[0],
                "age_group": key[1],
                "institution": key[2],
                "level": key[3],
                "faculty": key[4],
                "presentation_type": key[5],
                "experience": key[6],
                "devices": len(members),
                "recordings": recordings,
                "avg_confidence": round(avg_conf, 4),
                "top_emotion": top_emotions.most_common(1)[0][0],
            }
        )

    rows.sort(key=lambda r: (-r["recordings"], r["gender"], r["age_group"]))

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary_json = {
        "totals": {
            "devices": totals["devices"],
            "recordings": totals["recordings"],
            "avg_confidence": round(_safe_avg(totals["confidences"]), 4),
            "emotion_counts": dict(totals["emotion_counts"]),
        },
        "groups": rows,
    }
    json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    lines = [
        "# Session Analytics Summary",
        "",
        f"- Devices: {totals['devices']}",
        f"- Recordings: {totals['recordings']}",
        f"- Average confidence: {round(_safe_avg(totals['confidences']), 4)}",
        "",
        "## Top emotions (overall)",
    ]
    for emotion, count in totals["emotion_counts"].most_common():
        lines.append(f"- {emotion}: {count}")
    lines.append("")
    lines.append("## Group breakdown")
    if rows:
        lines.append(
            "| Gender | Age group | Institution | Level | Faculty | Presentation | Experience | Devices | Recordings | Avg confidence | Top emotion |"
        )
        lines.append(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        )
        for row in rows:
            lines.append(
                f"| {row['gender']} | {row['age_group']} | {row['institution']} | {row['level']} | {row['faculty']} | {row['presentation_type']} | {row['experience']} | {row['devices']} | {row['recordings']} | {row['avg_confidence']} | {row['top_emotion']} |"
            )
    else:
        lines.append("No session data found.")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    insights = []
    for row in rows:
        confidence = row["avg_confidence"]
        if confidence >= 0.85:
            confidence_level = "very high"
        elif confidence >= 0.7:
            confidence_level = "high"
        elif confidence >= 0.5:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        cohort = (
            f"{row['gender']} {row['institution']} students age {row['age_group']}"
        )
        insights.append(
            {
                "cohort": cohort,
                "confidence_level": confidence_level,
                "avg_confidence": confidence,
                "top_emotion": row["top_emotion"],
                "recordings": row["recordings"],
            }
        )

    insight_lines = ["# Cohort Insights", ""]
    if insights:
        for item in insights:
            insight_lines.append(
                f"- {item['cohort']} show {item['confidence_level']} confidence "
                f"(avg {item['avg_confidence']}) with top emotion {item['top_emotion']}."
            )
    else:
        insight_lines.append("No cohort insights available.")
    insights_path.write_text("\n".join(insight_lines), encoding="utf-8")
    insights_json_path.write_text(json.dumps(insights, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Aggregate session analytics data.")
    parser.add_argument(
        "--users-dir",
        default="results/users",
        help="Directory containing per-device session data.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/aggregates",
        help="Output directory for aggregated analytics.",
    )
    args = parser.parse_args()

    users_dir = Path(args.users_dir)
    if not users_dir.exists():
        print(f"No session data found at {users_dir}")
        return

    groups, totals = aggregate_sessions(users_dir)
    write_outputs(groups, totals, Path(args.out_dir))
    print("Wrote session analytics to results/session_summary.*")


if __name__ == "__main__":
    main()
