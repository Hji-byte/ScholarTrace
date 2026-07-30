from __future__ import annotations

import json
from pathlib import Path


CANDIDATES = Path("evaluation/results/dense-rrf-k15-top30-chunk-candidates.jsonl")
ANNOTATIONS = Path("evaluation/results/dense-rrf-k15-top30-chunk-codex-annotations.jsonl")
LABELS: dict[str, str] = {
    "csq001": "222222222222221222202202222202",
    "csq002": "220222222222221002220122222",
    "csq003": "222202222222202221212221212222",
    "csq004": "222222200222222222222222220222",
    "csq005": "020102202202022022022222202202",
    "csq006": "222222221222202222222222222022",
    "csq007": "002022200222222222022222202212",
    "csq008": "222022112222222222202022222111",
    "csq009": "002222220002220222222222020212",
    "csq010": "002002020200020022222022222202",
    "csq011": "222222222222220220222202202202",
    "csq012": "222220222222222222222222222222",
    "csq013": "202020022022022202222222022220",
    "csq014": "222110021222212222022022222221",
    "csq015": "222222222222222222221222212200",
    "csq016": "222222020222222202220202202022",
    "csq017": "221221202221222022222222220221",
    "csq018": "202222222022222220020200210220",
    "csq019": "222220221221222222222220202222",
    "csq020": "222220222222202202222222222202",
    "csq021": "121222200202221222222212222102",
    "csq022": "220222022020022221220010201200",
    "csq023": "220020020022021010222200222022",
    "csq024": "222222222022202222222002200222",
    "csq025": "222222222222222220222212221222",
    "csq026": "112222222220212000220120222122",
    "csq027": "202220022200122222022002020202",
    "csq028": "221202221022202122122020222022",
    "csq029": "222222222022122222022222222222",
    "csq030": "022222222022220220202220220222",
}

RATIONALES = {
    "2": "该段包含可直接回答问题的方法、指标、比较或研究发现。",
    "1": "该段提供相关背景或侧面信息，但没有直接回答问题核心。",
    "0": "该段为无关内容、泛泛提及或参考文献片段，不能作为回答证据。",
    "?": "文本提取或上下文不足，无法可靠判断。",
}


def main() -> None:
    candidates = [
        json.loads(line)
        for line in CANDIDATES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_question: dict[str, list[dict]] = {}
    for row in candidates:
        by_question.setdefault(row["question_id"], []).append(row)
    annotations = []
    for question_id, encoded in LABELS.items():
        rows = sorted(by_question[question_id], key=lambda row: row["rank"])
        if len(encoded) != len(rows):
            raise ValueError(f"{question_id}: {len(encoded)} labels for {len(rows)} chunks")
        for row, label in zip(rows, encoded, strict=True):
            if label not in RATIONALES:
                raise ValueError(f"{question_id}: invalid label {label}")
            annotations.append(
                {
                    "item_id": row["item_id"],
                    "question_id": question_id,
                    "rank": row["rank"],
                    "label": label,
                    "rationale": RATIONALES[label],
                    "annotator": "Codex",
                }
            )
    annotations.sort(key=lambda row: (row["question_id"], row["rank"]))
    ANNOTATIONS.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in annotations),
        encoding="utf-8",
    )
    print(f"Wrote {len(annotations)} Codex annotations across {len(LABELS)} questions.")


if __name__ == "__main__":
    main()
