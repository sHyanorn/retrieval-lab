package main

import (
	"bufio"
	"compress/gzip"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/dense"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

// synth writes the per-query measures of one model on one dataset, as
// cmd/eval -per-query writes them, in the layout collect reads. Neighbour
// overlap rises with ef, faster for later queries; nDCG@10 falls with the
// neighbours missed, by an amount that depends on the model and seed.
type synth struct {
	queries int
	shift   int     // added to every qid, to make qids differ
	loss    float64 // nDCG@10 lost per neighbour missed
	m8m32   bool    // write the exploratory builds
	anchor  bool
}

func (s synth) qids() []string {
	out := make([]string, s.queries)
	for i := range out {
		out[i] = fmt.Sprintf("q%03d", i+s.shift)
	}
	return out
}

func (s synth) exact() []eval.Query {
	var qs []eval.Query
	for i, id := range s.qids() {
		qs = append(qs, eval.Query{QID: id, NDCG10: 0.3 + 0.05*float64(i%7), Recall100: 0.5, Unjudged10: 0.1})
	}
	return qs
}

func (s synth) approx(seed, ef int) []eval.Query {
	qs := s.exact()
	for i := range qs {
		found := min(10, 6+(ef+5*i+seed)/12)
		q := &qs[i]
		q.NDCG10 -= s.loss * float64(10-found) * (1 + 0.1*float64(seed))
		q.Found10, q.Of10 = found, 10
		q.Unjudged10 = 0.1 + 0.01*float64(10-found)
		if ef < 100 {
			q.Recall100, q.Found100, q.Of100 = 0.1, found, 10
		} else {
			q.Found100, q.Of100 = 90+found, 100
		}
	}
	return qs
}

func (s synth) fused(seed int) []eval.Query {
	qs := s.exact()
	for i := range qs {
		qs[i].NDCG10 += 0.02 - 0.001*float64(seed)
	}
	return qs
}

func writeQueries(t *testing.T, path string, qs []eval.Query) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	w := bufio.NewWriter(f)
	if err := eval.WritePerQuery(w, qs); err != nil {
		t.Fatal(err)
	}
	if err := w.Flush(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func (s synth) write(t *testing.T, dir string) {
	t.Helper()
	writeQueries(t, filepath.Join(dir, "exact.tsv"), s.exact())
	builds := []build{{16, 200, 0}, {16, 200, 1}, {16, 200, 2}, {16, 200, 3}, {16, 200, 4}}
	if s.m8m32 {
		builds = append(builds, build{8, 200, 1}, build{32, 200, 1})
	}
	for _, b := range builds {
		for _, ef := range dense.StudyEFs {
			writeQueries(t, filepath.Join(dir, b.run(ef)+".tsv"), s.approx(b.seed+b.m, ef))
		}
	}
	if s.anchor {
		writeQueries(t, filepath.Join(dir, anchorBuild.run(anchorEF)+".tsv"), s.approx(0, anchorEF))
	}
	for _, k := range fusions {
		writeQueries(t, filepath.Join(dir, k+"-exact.tsv"), s.fused(0))
		for seed := 1; seed <= 4; seed++ {
			writeQueries(t, filepath.Join(dir, fmt.Sprintf("%s-s%d-ef100.tsv", k, seed)), s.fused(seed))
		}
	}
}

// readGz returns the lines of a stored file.
func readGz(t *testing.T, path string) []string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close() //nolint:errcheck // read-only
	z, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var lines []string
	sc := bufio.NewScanner(z)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	return lines
}

func TestCollect(t *testing.T) {
	in, out := t.TempDir(), t.TempDir()
	synth{queries: 6, loss: 0.01, m8m32: true, anchor: true}.write(t, filepath.Join(in, "bge-base-en-v1.5", "scifact"))
	synth{queries: 6, loss: 0.02}.write(t, filepath.Join(in, "gte-base", "scifact"))
	if err := collect(in, out, maxFile); err != nil {
		t.Fatal(err)
	}
	grid := strings.Split(strings.TrimSpace(readFile(t, filepath.Join(out, "grid.tsv"))), "\n")
	// 7 + 5 builds of 65 ef, and the anchor.
	if want := 1 + (7+5)*65 + 1; len(grid) != want {
		t.Fatalf("grid.tsv has %d lines, want %d", len(grid), want)
	}
	if grid[0] != gridHeader {
		t.Fatalf("grid header %q", grid[0])
	}
	// Seed 1 of bge-base, ef 10: the found counts are 6+(10+5i+17)/12, ...
	var at10, at100 string
	for _, l := range grid {
		if strings.HasPrefix(l, "bge-base-en-v1.5\tscifact\t16\t200\t1\t10\t") {
			at10 = l
		}
		if strings.HasPrefix(l, "bge-base-en-v1.5\tscifact\t16\t200\t1\t100\t") {
			at100 = l
		}
	}
	if !strings.HasSuffix(at10, "\t-") || strings.Count(at10, "/") != 1 {
		t.Fatalf("ef 10 row %q: want an exact recall@10 and no recall@100", at10)
	}
	if strings.HasSuffix(at100, "\t-") {
		t.Fatalf("ef 100 row %q: want recall@100", at100)
	}
	sel := strings.Split(strings.TrimSpace(readFile(t, filepath.Join(out, "select.tsv"))), "\n")
	if want := 1 + (7+5)*len(targets); len(sel) != want {
		t.Fatalf("select.tsv has %d lines, want %d", len(sel), want)
	}
	lines := readGz(t, filepath.Join(out, "scifact", "bge-base-en-v1.5.tsv.gz"))
	if lines[0] != storeHeader {
		t.Fatalf("store header %q", lines[0])
	}
	runs := map[string]int{}
	for _, l := range lines[1:] {
		f := strings.Split(l, "\t")
		runs[f[0]]++
		if strings.HasSuffix(f[0], "/ef10") && (f[3] != "-" || f[7] != "-" || f[8] != "-") {
			t.Fatalf("ef 10 row %q keeps measures at depth 100", l)
		}
	}
	for _, k := range []string{"exact", "m16-efc200-s0/ef40", "m16-efc200-s4/ef100", "m8-efc200-s1/ef10", "m32-efc200-s1/ef40", "m16-efc100-s0/ef1000", "rrf-exact", "norm-s4-ef100"} {
		if runs[k] != 6 {
			t.Errorf("run %s stored for %d queries, want 6", k, runs[k])
		}
	}
	if runs["m16-efc200-s1/ef1000"] != 0 {
		t.Error("a point no reading uses is stored")
	}
	// Seeds 2 to 4 are stored at seed 1's ef for 0.95 (H4).
	s1 := selectionOf(t, sel, "bge-base-en-v1.5", "16\t200\t1", "0.95")
	for seed := 2; seed <= 4; seed++ {
		if k := fmt.Sprintf("m16-efc200-s%d/ef%s", seed, s1); runs[k] != 6 {
			t.Errorf("run %s not stored", k)
		}
	}
	if got := readGz(t, filepath.Join(out, "scifact", "gte-base.tsv.gz")); len(got) < 2 {
		t.Fatal("second model not stored")
	}
}

func selectionOf(t *testing.T, sel []string, model, b, target string) string {
	t.Helper()
	for _, l := range sel {
		if strings.HasPrefix(l, model+"\tscifact\t"+b+"\t"+target+"\t") {
			return strings.Split(l, "\t")[6]
		}
	}
	t.Fatalf("no selection of %s %s at %s", model, b, target)
	return ""
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestCollectRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(t *testing.T, cell string)
		want   string
	}{
		{"qids of a run", func(t *testing.T, cell string) {
			writeQueries(t, filepath.Join(cell, "m16-efc200-s2", "ef40.tsv"), synth{queries: 6, shift: 1}.approx(2, 40))
		}, "qids"},
		{"fewer queries in a run", func(t *testing.T, cell string) {
			writeQueries(t, filepath.Join(cell, "rrf-s3-ef100.tsv"), synth{queries: 5}.fused(3))
		}, "qids"},
		{"missing ef", func(t *testing.T, cell string) { remove(t, filepath.Join(cell, "m16-efc200-s3", "ef17.tsv")) }, "missing"},
		{"missing build", func(t *testing.T, cell string) { removeAll(t, filepath.Join(cell, "m16-efc200-s0")) }, "missing"},
		{"missing exact", func(t *testing.T, cell string) { remove(t, filepath.Join(cell, "exact.tsv")) }, "missing"},
		{"missing fused run", func(t *testing.T, cell string) { remove(t, filepath.Join(cell, "norm-s2-ef100.tsv")) }, "missing"},
		{"ef off the grid", func(t *testing.T, cell string) {
			writeQueries(t, filepath.Join(cell, "m16-efc200-s1", "ef65.tsv"), synth{queries: 6}.approx(1, 65))
		}, "ef65"},
		{"unknown build", func(t *testing.T, cell string) {
			writeQueries(t, filepath.Join(cell, "m12-efc200-s1", "ef40.tsv"), synth{queries: 6}.approx(1, 40))
		}, "m12-efc200-s1"},
		{"seed 5", func(t *testing.T, cell string) {
			writeQueries(t, filepath.Join(cell, "m16-efc200-s5", "ef40.tsv"), synth{queries: 6}.approx(1, 40))
		}, "m16-efc200-s5"},
		{"stray file", func(t *testing.T, cell string) {
			writeQueries(t, filepath.Join(cell, "rrf-s0-ef100.tsv"), synth{queries: 6}.fused(0))
		}, "rrf-s0-ef100.tsv"},
		{"no neighbour overlap", func(t *testing.T, cell string) {
			writeQueries(t, filepath.Join(cell, "m16-efc200-s1", "ef40.tsv"), synth{queries: 6}.exact())
		}, "neighbour overlap"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			in := t.TempDir()
			cell := filepath.Join(in, "bge-base-en-v1.5", "scifact")
			synth{queries: 6, loss: 0.01}.write(t, cell)
			c.mutate(t, cell)
			err := collect(in, t.TempDir(), maxFile)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("collect: %v; want an error naming %q", err, c.want)
			}
		})
	}
}

func TestCollectModelsDisagree(t *testing.T) {
	in := t.TempDir()
	synth{queries: 6, loss: 0.01}.write(t, filepath.Join(in, "bge-base-en-v1.5", "scifact"))
	synth{queries: 6, shift: 1, loss: 0.01}.write(t, filepath.Join(in, "gte-base", "scifact"))
	if err := collect(in, t.TempDir(), maxFile); err == nil || !strings.Contains(err.Error(), "qids") {
		t.Fatalf("collect: %v; want an error on the qids of two models", err)
	}
}

func TestCollectLimit(t *testing.T) {
	in, out := t.TempDir(), t.TempDir()
	synth{queries: 6, loss: 0.01}.write(t, filepath.Join(in, "bge-base-en-v1.5", "scifact"))
	if err := collect(in, out, 200); err == nil || !strings.Contains(err.Error(), "limit") {
		t.Fatalf("collect: %v; want an error on the file size limit", err)
	}
	if _, err := os.Stat(filepath.Join(out, "scifact", "bge-base-en-v1.5.tsv.gz")); err == nil {
		t.Fatal("a file over the limit was written")
	}
}

func remove(t *testing.T, path string) {
	t.Helper()
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
}

func removeAll(t *testing.T, path string) {
	t.Helper()
	if err := os.RemoveAll(path); err != nil {
		t.Fatal(err)
	}
}

func writeGz(t *testing.T, path, body string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	z := gzip.NewWriter(f)
	if _, err := z.Write([]byte(body)); err != nil {
		t.Fatal(err)
	}
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}
