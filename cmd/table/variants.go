package main

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/bm25"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

// variantRuns are the runs beside the Lucene port's: the port with Bleve's
// differences switched on, by the -variant name of cmd/variants, and Bleve's
// own BM25 from make reference.
var variantRuns = []struct{ name, label string }{
	{"sqrt-tf", "V1 `sqrt-tf`: tf is sqrt(freq)"},
	{"avgdl", "V2 `avgdl`: Bleve's average length"},
	{"idf-twice", "V3 `idf-twice`: idf · queryNorm again"},
	{"coord", "V4 `coord`: coordination factor"},
	{"k1", "V5 `k1`: numerator times k1"},
	{"exact-length", "V6 `exact-length`: exact length"},
	{"bleve-length", "V6 `bleve-length`: Bleve's length"},
	{"all-docs", "V7 `all-docs`: idf over all documents"},
	{"all", "All: V1 to V5, V6 Bleve's length, V7"},
	{"bleve1", "Bleve's BM25, build 1"},
	{"bleve2", "Bleve's BM25, build 2"},
}

// variantsTables writes four tables, one column or row per dataset of the
// anchors, from RUNS/<dataset>.run (the Lucene port), the runs of
// variantRuns and RUNS/<dataset>.all.json (Bleve's statistics as
// cmd/variants prints them: documents and terms in the dictionary, whose
// ratio rounded up is Bleve's avgdl): nDCG@10 of each run and its
// difference from the port's; the number of positions at which each run's
// document differs from the port's; the run with all of Bleve's differences
// against Bleve's own runs; the statistics Lucene and Bleve score with.
func variantsTables(text string, anchors []anchor, data, runs string) (string, error) {
	ndcg := make([][]string, len(variantRuns)+1)
	lines := make([][]string, len(variantRuns)+1)
	var bleve, stats strings.Builder
	bleve.WriteString("| Dataset | nDCG@10 All | Bleve build 1 | build 2 | Lines that differ, All and build 1 | All and build 2 | build 1 and build 2 | Of these, scores more than 0.0001 apart: All and build 1 | All and build 2 | build 1 and build 2 |\n|---|---|---|---|---|---|---|---|---|---|\n")
	stats.WriteString("| Dataset | Documents | with a token | Tokens | avgdl, Lucene | Terms in the dictionary | avgdl, Bleve |\n|---|---|---|---|---|---|---|\n")
	for _, a := range anchors {
		qrels, err := beir.Qrels(filepath.Join(data, a.dataset), "test")
		if err != nil {
			return "", err
		}
		port, err := readRun(filepath.Join(runs, a.dataset+".run"))
		if err != nil {
			return "", err
		}
		base := eval.Mean(eval.Evaluate(qrels, port)).NDCG10
		ndcg[0] = append(ndcg[0], fmt.Sprintf("%.4f", base))
		lines[0] = append(lines[0], fmt.Sprint(runLines(port)))
		kept := map[string]eval.Run{}
		means := map[string]float64{}
		for i, v := range variantRuns {
			r, err := readRun(filepath.Join(runs, a.dataset+"."+v.name+".run"))
			if err != nil {
				return "", err
			}
			m := eval.Mean(eval.Evaluate(qrels, r)).NDCG10
			n, _ := differing(port, r)
			ndcg[i+1] = append(ndcg[i+1], fmt.Sprintf("%.4f (%s)", m, signed(delta(m, math.Round(base*1e4)/1e4))))
			lines[i+1] = append(lines[i+1], fmt.Sprint(n))
			if v.name == "all" || strings.HasPrefix(v.name, "bleve") {
				kept[v.name], means[v.name] = r, m
			}
		}
		var n, apart [3]int
		for i, p := range [3][2]string{{"all", "bleve1"}, {"all", "bleve2"}, {"bleve1", "bleve2"}} {
			n[i], apart[i] = differing(kept[p[0]], kept[p[1]])
		}
		fmt.Fprintf(&bleve, "| %s | %.4f | %.4f | %.4f | %d | %d | %d | %d | %d | %d |\n", a.dataset,
			means["all"], means["bleve1"], means["bleve2"], n[0], n[1], n[2], apart[0], apart[1], apart[2])

		var st struct {
			bm25.Stats
			Bleve struct {
				Docs  int64 `json:"docs"`
				Terms int64 `json:"terms"`
			} `json:"bleve"`
		}
		raw, err := os.ReadFile(filepath.Join(runs, a.dataset+".all.json"))
		if err == nil {
			err = json.Unmarshal(raw, &st)
		}
		if err != nil {
			return "", fmt.Errorf("%s statistics: %w", a.dataset, err)
		}
		fmt.Fprintf(&stats, "| %s | %d | %d | %d | %.2f | %d | %.0f |\n", a.dataset, st.Docs, st.NonEmpty, st.Tokens,
			float64(st.Tokens)/float64(st.NonEmpty), st.Bleve.Terms, math.Ceil(float64(st.Bleve.Terms)/float64(st.Bleve.Docs)))
	}
	var labels []string
	for _, v := range variantRuns {
		labels = append(labels, v.label)
	}
	tables := map[string]string{
		"variants-ndcg":  grid(anchors, append([]string{"Lucene BM25Similarity, ported"}, labels...), ndcg),
		"variants-lines": grid(anchors, append([]string{"Lines in the port's run"}, labels...), lines),
		"variants-bleve": bleve.String(),
		"variants-stats": stats.String(),
	}
	for _, name := range slices.Sorted(maps.Keys(tables)) {
		var err error
		if text, err = splice(text, name, tables[name]); err != nil {
			return "", err
		}
	}
	return text, nil
}

// grid is a table with one row per label and one column per dataset.
func grid(anchors []anchor, labels []string, cells [][]string) string {
	var b strings.Builder
	b.WriteString("| Run |")
	for _, a := range anchors {
		b.WriteString(" " + a.dataset + " |")
	}
	b.WriteString("\n|" + strings.Repeat("---|", len(anchors)+1) + "\n")
	for i, l := range labels {
		b.WriteString("| " + l + " | " + strings.Join(cells[i], " | ") + " |\n")
	}
	return b.String()
}

func runLines(r eval.Run) int {
	n := 0
	for _, hits := range r {
		n += len(hits)
	}
	return n
}

// differing counts the (query, rank) positions at which runs a and b hold
// different documents. Of these, apart counts the positions whose two
// documents' scores, rounded to four decimals as they were before the tie
// adjustment, are more than 0.0001 apart in a run that holds both; where
// neither run holds both, the two scores at the position are compared, and
// a position only one run has counts as apart.
func differing(a, b eval.Run) (n, apart int) {
	queries := maps.Clone(a)
	maps.Copy(queries, b)
	for q := range queries {
		x, y := a[q], b[q]
		rx, ry := rounded(x), rounded(y)
		for i := range max(len(x), len(y)) {
			if i < len(x) && i < len(y) && x[i].DocID == y[i].DocID {
				continue
			}
			n++
			if i >= len(x) || i >= len(y) {
				apart++
				continue
			}
			da, db := x[i].DocID, y[i].DocID
			var gap int64
			compared := false
			for _, r := range []map[string]int64{rx, ry} {
				sa, okA := r[da]
				sb, okB := r[db]
				if okA && okB {
					gap, compared = max(gap, abs(sa-sb)), true
				}
			}
			if !compared {
				gap = abs(rx[da] - ry[db])
			}
			if gap > 100 {
				apart++
			}
		}
	}
	return n, apart
}

// rounded undoes the tie adjustment (D007) on one query's hits: it returns
// each document's score rounded to four decimals, in units of 0.000001. A
// score not more than 0.0001 below the one before was lowered by 0.000001
// per tie so far, so adding that count back gives a multiple of 100.
func rounded(hits []eval.Hit) map[string]int64 {
	m := make(map[string]int64, len(hits))
	var prev, dup int64
	for i, h := range hits {
		p := int64(math.Round(h.Score * 1e6))
		if r := p + dup + 1; i > 0 && r%100 == 0 && prev-r <= 100 {
			dup++
		} else {
			dup = 0
		}
		m[h.DocID] = p + dup
		prev = p
	}
	return m
}

func abs(v int64) int64 { return max(v, -v) }
