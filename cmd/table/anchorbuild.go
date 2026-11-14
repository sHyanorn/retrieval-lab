package main

import (
	"bufio"
	"compress/gzip"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// The study's anchor build (Section 6 of the registration) and its model.
const (
	anchorModel = "BAAI_bge-base-en-v1.5"
	anchorRun   = "m16-efc100-s0/ef1000"
)

// anchorBuildTables writes the tables of amendment 24: the mean nDCG@10 of
// the anchor build and of exact search, from the per-query files of the
// study in DATA/<dataset>/<model>.tsv.gz, beside the Lucene HNSW numbers
// of results/anchors-hnsw.tsv, Pyserini 2.0.0's on every dataset and
// Lin's where the paper's table lists the dataset.
func anchorBuildTables(text, anchorsPath, study string) (string, error) {
	rows, err := readTSV(anchorsPath, "source\tdataset\tcondition\tndcg_cut_10\trecall_100\thnsw_avg_delta\thnsw_min_delta\thnsw_max_delta")
	if err != nil {
		return "", err
	}
	type means struct{ exact, anchor float64 }
	ours := map[string]means{}
	var p, l strings.Builder
	p.WriteString("| Dataset | exact | anchor build | Pyserini 2.0.0 | Δ |\n|---|---|---|---|---|\n")
	l.WriteString("| Dataset | exact | Lin: flat | exact minus anchor build | Lin: flat minus HNSW, mean [min, max] |\n|---|---|---|---|---|\n")
	for _, f := range rows {
		m, ok := ours[f[1]]
		if !ok {
			v, err := perQueryMeans(filepath.Join(study, f[1], anchorModel+".tsv.gz"), "exact", anchorRun)
			if err != nil {
				return "", err
			}
			m = means{v[0], v[1]}
			ours[f[1]] = m
		}
		switch f[0] {
		case "pyserini-2.0.0":
			pys, err := strconv.ParseFloat(f[3], 64)
			if err != nil {
				return "", fmt.Errorf("%s: %w", anchorsPath, err)
			}
			fmt.Fprintf(&p, "| %s | %.4f | %.4f | %.4f | %s |\n", f[1], m.exact, m.anchor, pys, signed(delta(m.anchor, pys)))
		case "lin-2025":
			fmt.Fprintf(&l, "| %s | %.4f | %s | %s | %s [%s, %s] |\n", f[1], m.exact, f[3], signed(m.exact-m.anchor), f[5], f[6], f[7])
		default:
			return "", fmt.Errorf("%s: unknown source %q", anchorsPath, f[0])
		}
	}
	if text, err = splice(text, "anchor-build-pyserini", p.String()); err != nil {
		return "", err
	}
	return splice(text, "anchor-build-lin", l.String())
}

// perQueryMeans gives the mean nDCG@10 of each named run in a per-query
// file of cmd/study collect; every run must cover the same number of
// queries.
func perQueryMeans(path string, runs ...string) ([]float64, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close() //nolint:errcheck // read-only
	gz, err := gzip.NewReader(fh)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	sum, n := make([]float64, len(runs)), make([]int, len(runs))
	sc := bufio.NewScanner(gz)
	for line := 0; sc.Scan(); line++ {
		f := strings.Split(sc.Text(), "\t")
		if line == 0 {
			if len(f) < 3 || f[0] != "run" || f[2] != "ndcg_cut_10" {
				return nil, fmt.Errorf("%s: header %q", path, sc.Text())
			}
			continue
		}
		i := slices.Index(runs, f[0])
		if i < 0 {
			continue
		}
		v, err := strconv.ParseFloat(f[2], 64)
		if err != nil {
			return nil, fmt.Errorf("%s line %d: %w", path, line+1, err)
		}
		sum[i] += v
		n[i]++
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	for i := range runs {
		if n[i] == 0 || n[i] != n[0] {
			return nil, fmt.Errorf("%s: %d queries of run %s, %d of %s", path, n[i], runs[i], n[0], runs[0])
		}
		sum[i] /= float64(n[i])
	}
	return sum, nil
}
