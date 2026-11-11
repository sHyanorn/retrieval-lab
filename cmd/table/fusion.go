package main

import (
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

// fusionTables writes the fusion tables (SPEC §4, D015) from
// RUNS/<dataset>.{run,bge.run,rrf.run,norm.run}: nDCG@10 of the two fused
// runs against the anchors, and the difference of each from its two
// inputs with a paired bootstrap interval. A dataset without fused runs is
// listed as not run.
func fusionTables(text, anchorsPath, data, runs string) (string, error) {
	rows, err := readTSV(anchorsPath, "dataset\tpyserini_rrf\tpyserini_normalize\tanserini_rrf\tanserini_normalize\tpage_bge")
	if err != nil {
		return "", err
	}
	var a, g strings.Builder
	a.WriteString("| Dataset | RRF Pyserini | Anserini | here | Δ | Normalize Pyserini | Anserini | here | Δ | BGE on the page | BGE here | agreement, RRF / Normalize |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
	g.WriteString("| Dataset | BM25 | BGE | RRF − BM25 | RRF − BGE | Normalize − BM25 | Normalize − BGE |\n|---|---|---|---|---|---|---|\n")
	for _, f := range rows {
		var anchor [5]float64
		for i := range anchor {
			if anchor[i], err = strconv.ParseFloat(f[i+1], 64); err != nil {
				return "", fmt.Errorf("%s: %w", anchorsPath, err)
			}
		}
		qrels, err := beir.Qrels(filepath.Join(data, f[0]), "test")
		if err != nil {
			return "", err
		}
		var per [4]map[string]eval.Scores // BM25, BGE, RRF, Normalize
		for i, suffix := range []string{".run", ".bge.run", ".rrf.run", ".norm.run"} {
			r, err := readRun(filepath.Join(runs, f[0]+suffix))
			if errors.Is(err, fs.ErrNotExist) {
				break
			}
			if err != nil {
				return "", err
			}
			per[i] = eval.Evaluate(qrels, r)
		}
		if per[3] == nil {
			fmt.Fprintf(&a, "| %s | not run |\n", f[0])
			fmt.Fprintf(&g, "| %s | not run |\n", f[0])
			continue
		}
		var mean [4]float64
		for i, p := range per {
			mean[i] = eval.Mean(p).NDCG10
		}
		dr, dn := delta(mean[2], anchor[0]), delta(mean[3], anchor[1])
		fmt.Fprintf(&a, "| %s | %.4f | %.4f | %.4f | %s | %.4f | %.4f | %.4f | %s | %.4f | %.4f | %s / %s |\n", f[0],
			anchor[0], anchor[2], mean[2], signed(dr), anchor[1], anchor[3], mean[3], signed(dn), anchor[4], mean[1], agreement(dr), agreement(dn))
		fmt.Fprintf(&g, "| %s | %.4f | %.4f | %s | %s | %s | %s |\n", f[0], mean[0], mean[1],
			gain(per[2], per[0]), gain(per[2], per[1]), gain(per[3], per[0]), gain(per[3], per[1]))
	}
	if text, err = splice(text, "fusion", a.String()); err != nil {
		return "", err
	}
	return splice(text, "fusion-gain", g.String())
}

// gain is the mean nDCG@10 of a minus that of b over the queries of the
// qrels, with its paired bootstrap interval.
func gain(a, b map[string]eval.Scores) string {
	var x, y []float64
	for _, qid := range slices.Sorted(maps.Keys(a)) {
		x, y = append(x, a[qid].NDCG10), append(y, b[qid].NDCG10)
	}
	lo, hi := eval.PairedInterval(x, y, 10000)
	return fmt.Sprintf("%+.4f [%+.4f, %+.4f]", eval.Mean(a).NDCG10-eval.Mean(b).NDCG10, lo, hi)
}
