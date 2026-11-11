package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/dense"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

// annTables writes the tables of the HNSW runs (SPEC §4, D014), one row per
// dataset of the BGE anchors: recall@10 and recall@100 of each
// RUNS/<dataset>.bge-hnsw-ef<ef>.run against RUNS/<dataset>.bge.run, the
// change in nDCG@10, and the search times from RUNS/<dataset>.ann.json. A
// dataset without the times file is listed as not run.
func annTables(text, anchorsPath, data, runs string) (string, error) {
	anchors, err := readDenseAnchors(anchorsPath)
	if err != nil {
		return "", err
	}
	var deep []int
	for _, ef := range dense.EFs {
		if dense.GraphDepth(ef) >= 100 {
			deep = append(deep, ef)
		}
	}
	r10, r100, nd, tm := header("Dataset", dense.EFs), header("Dataset", deep), header("Dataset | exact", dense.EFs), header("Dataset | build, s | exact", dense.EFs)
	for _, a := range anchors {
		var t struct {
			Build float64 `json:"build_seconds"`
			Exact struct {
				Mean float64 `json:"mean_ms"`
			} `json:"exact"`
			HNSW []struct {
				Mean float64 `json:"mean_ms"`
			} `json:"hnsw"`
		}
		raw, err := os.ReadFile(filepath.Join(runs, a.dataset+".ann.json"))
		if errors.Is(err, fs.ErrNotExist) {
			for _, b := range []*strings.Builder{r10, r100, nd, tm} {
				fmt.Fprintf(b, "| %s | not run |\n", a.dataset)
			}
			continue
		}
		if err == nil {
			err = json.Unmarshal(raw, &t)
		}
		if err == nil && len(t.HNSW) != len(dense.EFs) {
			err = fmt.Errorf("%d HNSW timings for %d ef values", len(t.HNSW), len(dense.EFs))
		}
		if err != nil {
			return "", fmt.Errorf("%s timings: %w", a.dataset, err)
		}
		exact, err := readRun(filepath.Join(runs, a.dataset+".bge.run"))
		if err != nil {
			return "", err
		}
		qrels, err := beir.Qrels(filepath.Join(data, a.dataset), "test")
		if err != nil {
			return "", err
		}
		exactNDCG := eval.Mean(eval.Evaluate(qrels, exact)).NDCG10
		fmt.Fprintf(r10, "| %s |", a.dataset)
		fmt.Fprintf(r100, "| %s |", a.dataset)
		fmt.Fprintf(nd, "| %s | %.4f |", a.dataset, exactNDCG)
		for _, ef := range dense.EFs {
			approx, err := readRun(filepath.Join(runs, fmt.Sprintf("%s.bge-hnsw-ef%d.run", a.dataset, ef)))
			if err != nil {
				return "", err
			}
			fmt.Fprintf(r10, " %.4f |", overlap(exact, approx, 10))
			if dense.GraphDepth(ef) >= 100 {
				fmt.Fprintf(r100, " %.4f |", overlap(exact, approx, 100))
			}
			fmt.Fprintf(nd, " %s |", ndcgChange(eval.Mean(eval.Evaluate(qrels, approx)).NDCG10, exactNDCG))
		}
		for _, b := range []*strings.Builder{r10, r100, nd} {
			b.WriteString("\n")
		}
		fmt.Fprintf(tm, "| %s | %.0f | %.2f |", a.dataset, t.Build, t.Exact.Mean)
		for _, h := range t.HNSW {
			fmt.Fprintf(tm, " %.2f |", h.Mean)
		}
		tm.WriteString("\n")
	}
	for _, s := range []struct {
		name string
		b    *strings.Builder
	}{{"hnsw-recall10", r10}, {"hnsw-recall100", r100}, {"hnsw-ndcg", nd}, {"hnsw-time", tm}} {
		if text, err = splice(text, s.name, s.b.String()); err != nil {
			return "", err
		}
	}
	return text, nil
}

func header(first string, efs []int) *strings.Builder {
	var b strings.Builder
	b.WriteString("| " + first + " |")
	cols := strings.Count(first, "|") + 1
	for _, ef := range efs {
		fmt.Fprintf(&b, " ef %d |", ef)
	}
	b.WriteString("\n|" + strings.Repeat("---|", cols+len(efs)) + "\n")
	return &b
}

// overlap averages eval.Overlap over the queries of the exact run.
func overlap(exact, approx eval.Run, k int) float64 {
	var sum float64
	for qid, hits := range exact {
		sum += eval.Overlap(hits, approx[qid], k)
	}
	return sum / float64(len(exact))
}

// ndcgChange formats approx − exact to four decimals.
func ndcgChange(approx, exact float64) string { return signed(approx - exact) }
