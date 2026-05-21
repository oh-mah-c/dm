import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_dm():
    subprocess.run(["make", "bin/dm.exe"], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def write_reps_example(tmp_path: Path):
    dfa = tmp_path / "reps_example.dfa"
    dfa.write_text(
        "\n".join(
            [
                "states 8",
                "start 0",
                "final 3 1 ABC",
                "final 7 2 ABCD",
                "trans 0 a 1",
                "trans 0 d 7",
                "trans 1 b 2",
                "trans 2 c 3",
                "trans 3 a 4",
                "trans 3 d 7",
                "trans 4 b 5",
                "trans 5 c 6",
                "trans 6 a 4",
                "trans 6 d 7",
            ]
        )
        + "\n"
    )
    return dfa


def run_dm(*args):
    return subprocess.run([str(ROOT / "bin" / "dm.exe"), *args], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_reps_linear_table_pathology(tmp_path):
    build_dm()
    dfa = write_reps_example(tmp_path)
    text = tmp_path / "input.txt"
    text.write_text("abcabcabcabc")

    out = run_dm("maximal_munch", "--dfa", str(dfa), "--input", str(text)).stdout.strip().splitlines()
    assert out == [
        "ABC\t1\t0\t3\tabc",
        "ABC\t1\t3\t6\tabc",
        "ABC\t1\t6\t9\tabc",
        "ABC\t1\t9\t12\tabc",
    ]

    stats = dict(line.split("\t", 1) for line in run_dm("maximal_munch", "--dfa", str(dfa), "--input", str(text), "--stats").stdout.strip().splitlines())
    assert stats["tokens"] == "4"
    assert stats["errors"] == "0"
    assert stats["tab_states"] == "3"
    assert int(stats["failed_hits"]) > 0
    assert int(stats["failed_marks"]) > 0


def test_reps_munch_reports_scanner_errors(tmp_path):
    build_dm()
    dfa = write_reps_example(tmp_path)
    text = tmp_path / "bad.txt"
    text.write_text("abx")

    out = run_dm("maximal_munch", "--dfa", str(dfa), "--input", str(text)).stdout.strip().splitlines()
    assert out == ["ERROR\t0\t1\ta", "ERROR\t1\t2\tb", "ERROR\t2\t3\tx"]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        test_reps_linear_table_pathology(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_reps_munch_reports_scanner_errors(Path(td))
    print("maximal_munch tests passed")
