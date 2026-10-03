import csv
from pathlib import Path

def inspect_eval_data():
    project_root = Path(__file__).resolve().parent.parent
    test_csv = project_root / "data" / "evaluation" / "test.csv"
    sample_sub_csv = project_root / "submission" / "sample_submission.csv"
    
    print("=" * 60)
    print("INSPECTING test.csv")
    print("=" * 60)
    with open(test_csv, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        print(f"Total questions found: {len(reader)}")
        print(f"Column headers: {list(reader[0].keys())}")
        print("\nAll Questions:")
        for row in reader:
            qid = row["question_id"]
            q_text = row["question"]
            category = "In-Scope (HR Policy)" if int(qid[1:]) <= 15 else "Out-of-Scope / Edge-Case"
            print(f"[{qid}] ({category}): {q_text}")
            
    print("\n" + "=" * 60)
    print("INSPECTING sample submission.csv")
    print("=" * 60)
    with open(sample_sub_csv, mode="r", encoding="utf-8") as f:
        sub_reader = list(csv.DictReader(f))
        print(f"Total submission rows: {len(sub_reader)}")
        print(f"Column headers: {list(sub_reader[0].keys())}")
        print("First 3 sample rows:")
        for row in sub_reader[:3]:
            print(f"  {row}")

if __name__ == "__main__":
    inspect_eval_data()
