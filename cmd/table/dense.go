package main

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/dense"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

type denseAnchor struct {
	dataset                                      string
	flatNDCG, flatRecall, faissNDCG, faissRecall float64
}

// denseTables writes the BGE-base table from RUNS/<dataset>.bge.run and the
// vector agreement table from VECTORS/<dataset> against PREBUILT/<dataset>.
// A dataset without a run file is listed as not run. It also returns what
// falls short of SPEC §4 and D012: an nDCG@10 more than 0.005 from the flat
// anchor, a document whose cosine to Pyserini's vector is below 0.9999.
func denseTables(text, anchorsPath, runs, vectors, prebuilt string, mean func(string, string) (eval.Scores, error)) (string, []string, error) {
	anchors, err := readDenseAnchors(anchorsPath)
	if err != nil {
		return "", nil, err
	}
	var short []string
	var t, v strings.Builder
	t.WriteString("| Dataset | nDCG@10 Pyserini flat | Pyserini faiss | here | Δ to flat | R@100 Pyserini flat | here | Δ to flat | nDCG@10 agreement |\n|---|---|---|---|---|---|---|---|---|\n")
	v.WriteString("| Dataset | Documents | Min cosine | 0.1 % | 1 % | Median | Below 0.9999 | Below 0.99999 |\n|---|---|---|---|---|---|---|---|\n")
	for _, a := range anchors {
		m, err := mean(a.dataset, a.dataset+".bge.run")
		if errors.Is(err, fs.ErrNotExist) {
			fmt.Fprintf(&t, "| %s | %.4f | %.4f | not run | | %.4f | not run | | |\n", a.dataset, a.flatNDCG, a.faissNDCG, a.flatRecall)
			continue
		}
		if err != nil {
			return "", nil, err
		}
		dn, dr := delta(m.NDCG10, a.flatNDCG), delta(m.Recall100, a.flatRecall)
		if agreement(dn) == "outside 0.005" {
			short = append(short, fmt.Sprintf("%s: nDCG@10 %.4f, %s from the flat anchor", a.dataset, m.NDCG10, signed(dn)))
		}
		fmt.Fprintf(&t, "| %s | %.4f | %.4f | %.4f | %s | %.4f | %.4f | %s | %s |\n",
			a.dataset, a.flatNDCG, a.faissNDCG, m.NDCG10, signed(dn), a.flatRecall, m.Recall100, signed(dr), agreement(dn))

		theirs := filepath.Join(prebuilt, a.dataset)
		if _, err := os.Stat(theirs); errors.Is(err, fs.ErrNotExist) {
			continue
		}
		g, err := agreementOf(filepath.Join(vectors, a.dataset), theirs)
		if err != nil {
			return "", nil, err
		}
		if g.Below9999 > 0 {
			short = append(short, fmt.Sprintf("%s: %d documents with a cosine below 0.9999, the lowest %.6f", a.dataset, g.Below9999, g.Min))
		}
		fmt.Fprintf(&v, "| %s | %d | %.6f | %.6f | %.6f | %.7f | %d | %d |\n",
			a.dataset, g.Docs, g.Min, g.P001, g.P01, g.P50, g.Below9999, g.Below99999)
	}
	if text, err = splice(text, "bge-flat", t.String()); err == nil {
		text, err = splice(text, "bge-vectors", v.String())
	}
	return text, short, err
}

func agreementOf(ours, theirs string) (dense.Agreement, error) {
	aIDs, a, dim, err := dense.ReadFaissFlat(ours)
	if err != nil {
		return dense.Agreement{}, err
	}
	bIDs, b, _, err := dense.ReadFaissFlat(theirs)
	if err != nil {
		return dense.Agreement{}, err
	}
	return dense.Compare(aIDs, a, bIDs, b, dim)
}

func readDenseAnchors(path string) ([]denseAnchor, error) {
	rows, err := readTSV(path, "dataset\tflat_ndcg_cut_10\tflat_recall_100\tfaiss_ndcg_cut_10\tfaiss_recall_100")
	if err != nil {
		return nil, err
	}
	out := make([]denseAnchor, len(rows))
	for n, f := range rows {
		a := denseAnchor{dataset: f[0]}
		for i, p := range []*float64{&a.flatNDCG, &a.flatRecall, &a.faissNDCG, &a.faissRecall} {
			if *p, err = strconv.ParseFloat(f[i+1], 64); err != nil {
				return nil, fmt.Errorf("%s line %d: %w", path, n+2, err)
			}
		}
		out[n] = a
	}
	return out, nil
}
