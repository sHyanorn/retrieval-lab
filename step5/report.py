"""The small files of step 5's output, carried out of the pod by its log.

encode/report.py's format and reader, over step 5's files.

usage: python3 step5/report.py OUT              print OUT's files
       python3 step5/report.py --read LOG DIR   check the last report in LOG, write its files under DIR
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "encode"))
import report as R  # noqa: E402

R.FILES = ("summary.json", "check3.tsv", "runs.sha256", "perquery.sha256")

if __name__ == "__main__":
    sys.exit(R.main(sys.argv[1:]))
