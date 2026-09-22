import sys
import json
from citeguard.pipeline import run_pipeline
from citeguard.aggregator import to_text_report

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_manuscript.txt"

    with open(path) as f:
        text = f.read()

    report = run_pipeline(text)

    print(to_text_report(report))

    with open("output_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("full report saved to output_report.json")
