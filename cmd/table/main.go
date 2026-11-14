// Command table computes results from run files and writes them into
// docs/BENCHMARKS.md, between marker comments, next to the anchors. By
// default it writes the BM25 tables from <dataset>.run and
// <dataset>.stats.json in -runs; with -reference, the Bleve reference table
// from <dataset>.bleve1.run and <dataset>.bleve2.run; with -dense, the
// BGE-base tables from <dataset>.bge.run and the vectors in -vectors and
// -prebuilt.
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/bm25"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

type anchor struct {
	dataset      string
	ndcg, recall float64
	docs, tokens int64
}

func main() {
	anchorsPath := flag.String("anchors", "results/anchors.tsv", "anchors file (results/anchors-bge.tsv with -dense and -ann)")
	data := flag.String("data", "", "directory holding the BEIR datasets")
	runs := flag.String("runs", "", "directory holding the run files and index statistics")
	doc := flag.String("doc", "docs/BENCHMARKS.md", "file to update")
	reference := flag.Bool("reference", false, "write the Bleve reference table")
	denseMode := flag.Bool("dense", false, "write the BGE-base tables")
	annMode := flag.Bool("ann", false, "write the HNSW tables")
	fusionMode := flag.Bool("fusion", false, "write the fusion tables (with -anchors results/anchors-fusion.tsv)")
	anchorBuild := flag.Bool("anchor-build", false, "write the tables of the study's anchor build from -data (with -anchors results/anchors-hnsw.tsv and -data study/results)")
	vectors := flag.String("vectors", "", "with -dense: directory of this project's document vectors, one per dataset")
	prebuilt := flag.String("prebuilt", "", "with -dense: directory of Pyserini's flat indexes, one per dataset")
	check := flag.Bool("check", false, "with -dense: write nothing; exit 1 if a result falls short of the tolerance or a vector of the agreement threshold")
	flag.Parse()
	if err := run(*anchorsPath, *data, *runs, *doc, *reference, *denseMode, *annMode, *fusionMode, *anchorBuild, *check, *vectors, *prebuilt); err != nil {
		fmt.Fprintln(os.Stderr, "table:", err)
		os.Exit(1)
	}
}

func run(anchorsPath, data, runs, doc string, reference, denseMode, annMode, fusionMode, anchorBuild, check bool, vectors, prebuilt string) error {
	buf, err := os.ReadFile(doc)
	if err != nil {
		return err
	}
	text := string(buf)
	mean := func(dataset, file string) (eval.Scores, error) {
		return meanOf(filepath.Join(data, dataset), filepath.Join(runs, file))
	}
	var short []string
	switch {
	case annMode:
		text, err = annTables(text, anchorsPath, data, runs)
	case fusionMode:
		text, err = fusionTables(text, anchorsPath, data, runs)
	case anchorBuild:
		text, err = anchorBuildTables(text, anchorsPath, data)
	case denseMode:
		text, short, err = denseTables(text, anchorsPath, runs, vectors, prebuilt, mean)
	default:
		var anchors []anchor
		if anchors, err = readAnchors(anchorsPath); err == nil && reference {
			text, err = referenceTable(text, anchors, mean)
		} else if err == nil {
			text, err = mainTables(text, anchors, runs, mean)
		}
	}
	if err != nil {
		return err
	}
	if check {
		if len(short) > 0 {
			return fmt.Errorf("outside the acceptance:\n%s", strings.Join(short, "\n"))
		}
		return nil
	}
	return os.WriteFile(doc, []byte(text), 0o644) //nolint:gosec // a tracked text file
}

func mainTables(text string, anchors []anchor, runs string, mean func(string, string) (eval.Scores, error)) (string, error) {
	var t, s strings.Builder
	t.WriteString("| Dataset | nDCG@10 Pyserini | nDCG@10 here | Δ | R@100 Pyserini | R@100 here | Δ | nDCG@10 agreement |\n|---|---|---|---|---|---|---|---|\n")
	s.WriteString("| Dataset | Non-empty documents, Anserini | here | Tokens, Anserini | here | Δ tokens |\n|---|---|---|---|---|---|\n")
	for _, a := range anchors {
		m, err := mean(a.dataset, a.dataset+".run")
		if err != nil {
			return "", err
		}
		dn, dr := delta(m.NDCG10, a.ndcg), delta(m.Recall100, a.recall)
		fmt.Fprintf(&t, "| %s | %.4f | %.4f | %s | %.4f | %.4f | %s | %s |\n",
			a.dataset, a.ndcg, m.NDCG10, signed(dn), a.recall, m.Recall100, signed(dr), agreement(dn))

		var st bm25.Stats
		raw, err := os.ReadFile(filepath.Join(runs, a.dataset+".stats.json"))
		if err == nil {
			err = json.Unmarshal(raw, &st)
		}
		if err != nil {
			return "", fmt.Errorf("%s statistics: %w", a.dataset, err)
		}
		fmt.Fprintf(&s, "| %s | %d | %d | %d | %d | %+d (%+.4f%%) |\n", a.dataset, a.docs, st.NonEmpty,
			a.tokens, st.Tokens, st.Tokens-a.tokens, 100*float64(st.Tokens-a.tokens)/float64(a.tokens))
	}
	text, err := splice(text, "bm25-flat", t.String())
	if err != nil {
		return "", err
	}
	return splice(text, "index-stats", s.String())
}

func referenceTable(text string, anchors []anchor, mean func(string, string) (eval.Scores, error)) (string, error) {
	var t strings.Builder
	t.WriteString("| Dataset | nDCG@10 Pyserini | Bleve BM25, build 1 | build 2 | Δ, larger |\n|---|---|---|---|---|\n")
	for _, a := range anchors {
		m1, err := mean(a.dataset, a.dataset+".bleve1.run")
		if err != nil {
			return "", err
		}
		m2, err := mean(a.dataset, a.dataset+".bleve2.run")
		if err != nil {
			return "", err
		}
		d1, d2 := delta(m1.NDCG10, a.ndcg), delta(m2.NDCG10, a.ndcg)
		if math.Abs(d2) > math.Abs(d1) {
			d1 = d2
		}
		fmt.Fprintf(&t, "| %s | %.4f | %.4f | %.4f | %s |\n", a.dataset, a.ndcg, m1.NDCG10, m2.NDCG10, signed(d1))
	}
	return splice(text, "bleve-bm25", t.String())
}

// delta is the difference to the anchor of the value as trec_eval prints
// it, to four decimals.
func delta(v, anchor float64) float64 { return math.Round(v*1e4)/1e4 - anchor }

func signed(d float64) string {
	if math.Abs(d) < 5e-5 {
		return "0.0000"
	}
	return fmt.Sprintf("%+.4f", d)
}

// agreement classifies an nDCG@10 difference against SPEC §4.
func agreement(d float64) string {
	switch d = math.Abs(d); {
	case d < 5e-5:
		return "all four decimals"
	case d <= 0.0005+5e-5:
		return "three printed decimals"
	case d <= 0.005+5e-5:
		return "within 0.005"
	default:
		return "outside 0.005"
	}
}

func meanOf(dataDir, runPath string) (eval.Scores, error) {
	qrels, err := beir.Qrels(dataDir, "test")
	if err != nil {
		return eval.Scores{}, err
	}
	r, err := readRun(runPath)
	if err != nil {
		return eval.Scores{}, err
	}
	return eval.Mean(eval.Evaluate(qrels, r)), nil
}

func readRun(path string) (eval.Run, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close() //nolint:errcheck // read-only
	r, err := eval.ReadRun(bufio.NewReader(f))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return r, nil
}

// splice replaces the text between "<!-- name -->" and "<!-- /name -->".
func splice(text, name, table string) (string, error) {
	start, end := "<!-- "+name+" -->\n", "<!-- /"+name+" -->"
	i, j := strings.Index(text, start), strings.Index(text, end)
	if i < 0 || j < i {
		return "", fmt.Errorf("markers for %s not found", name)
	}
	return text[:i+len(start)] + table + text[j:], nil
}

func readAnchors(path string) ([]anchor, error) {
	rows, err := readTSV(path, "dataset\tndcg_cut_10\trecall_100\tdocs_non_empty\ttokens")
	if err != nil {
		return nil, err
	}
	out := make([]anchor, len(rows))
	for n, f := range rows {
		a := anchor{dataset: f[0]}
		var errs [4]error
		a.ndcg, errs[0] = strconv.ParseFloat(f[1], 64)
		a.recall, errs[1] = strconv.ParseFloat(f[2], 64)
		a.docs, errs[2] = strconv.ParseInt(f[3], 10, 64)
		a.tokens, errs[3] = strconv.ParseInt(f[4], 10, 64)
		if err := errors.Join(errs[:]...); err != nil {
			return nil, fmt.Errorf("%s line %d: %w", path, n+2, err)
		}
		out[n] = a
	}
	return out, nil
}

// readTSV reads a tab-separated file whose first line must be header and
// whose other lines have as many fields.
func readTSV(path, header string) ([][]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if lines[0] != header {
		return nil, fmt.Errorf("%s: header %q, want %q", path, lines[0], header)
	}
	width := len(strings.Split(header, "\t"))
	rows := make([][]string, 0, len(lines)-1)
	for n, line := range lines[1:] {
		f := strings.Split(line, "\t")
		if len(f) != width {
			return nil, fmt.Errorf("%s line %d: want %d fields", path, n+2, width)
		}
		rows = append(rows, f)
	}
	return rows, nil
}
