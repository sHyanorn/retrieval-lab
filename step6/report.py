"""The files of step 6's output, carried out of the pod by its log.

encode/report.py's format and reader, over VOL/step6/out. Every file that
holds a label-based measure, cmd/study collect's grid.tsv and per-query files
and step 5's per-query files, is printed in base64 only, so that no measure
stands as text in the log; the others are label-free and printed as text.

usage: python3 step6/report.py OUT              print OUT's files
       python3 step6/report.py --read LOG DIR   check the last report in LOG, write its files under DIR
"""

import importlib.util
import os
import sys

# encode/report.py under a name of its own, so that this module, also named report, can import it.
_spec = importlib.util.spec_from_file_location(
    "encode_report", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "encode", "report.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

R.FILES = ("summary.json", "environment.json", "units.tsv", "neighbour.tsv", "runs.sha256", "perquery.sha256",
           "dense-times.jsonl", "results/select.tsv", "results/per-query.sha256", "results/grid.tsv", "results/*/*.tsv.gz",
           "step5/study/*/*/exact.tsv", "step5/bm25/*.tsv")
R.BINARY = ("results/grid.tsv", "results/*/*.tsv.gz", "step5/*")

if __name__ == "__main__":
    sys.exit(R.main(sys.argv[1:]))
