package main

import (
	"math"
	"math/big"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/study"
)

// mem builds results in memory: each cell's runs hold the nDCG@10 given,
// and each confirmatory build's grid has its recall at the ef given.
type mem struct{ r *results }

func newMem() mem {
	return mem{&results{cells: map[string]map[string]*cell{}, grid: map[gridKey][]study.Point{}}}
}

func (m mem) cell(model, ds string, qids []string, runs map[string][]float64) {
	if m.r.cells[ds] == nil {
		m.r.cells[ds] = map[string]*cell{}
	}
	c := &cell{qids: qids, runs: map[string][]row{}}
	for run, v := range runs {
		_, ef, cut := strings.Cut(run, "/ef")
		deep := !cut || ef >= "100" && len(ef) >= 3
		for _, x := range v {
			c.runs[run] = append(c.runs[run], row{ndcg: x, recall100: x, deep: deep})
		}
	}
	m.r.cells[ds][model] = c
}

func (m mem) recall(model, ds string, b build, recall map[int]string) {
	for ef, rec := range recall {
		r, _ := new(big.Rat).SetString(rec)
		m.r.grid[gridKey{model, ds, b}] = append(m.r.grid[gridKey{model, ds, b}], study.Point{EF: ef, Recall: r})
	}
}

func TestLossAveragesBuilds(t *testing.T) {
	m := newMem()
	runs := map[string][]float64{"exact": {0.5, 0.6}}
	for s := 1; s <= 4; s++ {
		runs[confirmatory(s).run(40)] = []float64{0.5 - 0.01*float64(s), 0.6}
		// At recall 0.9 seed s is read at ef 10 + s, where it loses s/100
		// on the second query.
		ef := 10 + s
		runs[confirmatory(s).run(ef)] = []float64{0.5, 0.6 - 0.01*float64(s)}
		m.recall("a", "scifact", confirmatory(s), map[int]string{ef: "0.9", ef - 1: "0.89", ef + 1: "0.92"})
	}
	m.cell("a", "scifact", []string{"q1", "q2"}, runs)
	loss, err := m.r.loss([]string{"a"}, []string{"scifact"}, atEF(40))
	if err != nil {
		t.Fatal(err)
	}
	if got := []float64{loss[0][0][0], loss[0][1][0]}; math.Abs(got[0]-0.025) > 1e-12 || got[1] != 0 {
		t.Fatalf("loss at ef 40 = %v, want [0.025 0]", got)
	}
	loss, err = m.r.loss([]string{"a"}, []string{"scifact"}, atRecall("0.90", false))
	if err != nil {
		t.Fatal(err)
	}
	// Each build at its own ef: seed 1's ef for all would give 0.01.
	if got := loss[0][1][0]; math.Abs(got-0.025) > 1e-12 {
		t.Fatalf("loss at recall 0.90 = %v, want 0.025, each build at its own ef", got)
	}
	approx, err := m.r.approx([]string{"a"}, []string{"scifact"}, atRecall("0.90", false))
	if err != nil {
		t.Fatal(err)
	}
	if got := approx[0][1][0]; math.Abs(got-0.575) > 1e-12 {
		t.Fatalf("approx at recall 0.90 = %v, want 0.575", got)
	}
}

func TestStrataRejects(t *testing.T) {
	m := newMem()
	runs := func() map[string][]float64 {
		r := map[string][]float64{"exact": {0.5, 0.6}}
		for s := 1; s <= 4; s++ {
			r[confirmatory(s).run(40)] = []float64{0.4, 0.5}
			r[confirmatory(s).run(10)] = []float64{0.4, 0.5}
		}
		return r
	}
	m.cell("a", "scifact", []string{"q1", "q2"}, runs())
	m.cell("b", "scifact", []string{"q1", "q3"}, runs())
	short := runs()
	delete(short, confirmatory(3).run(40))
	m.cell("c", "scifact", []string{"q1", "q2"}, short)
	for _, c := range []struct {
		models []string
		p      point
		want   string
	}{
		{[]string{"a", "b"}, atEF(40), "qids of b"},
		{[]string{"a", "c"}, atEF(40), "run m16-efc200-s3/ef40 not stored"},
		{[]string{"a", "d"}, atEF(40), "d on scifact: not stored"},
	} {
		if _, err := m.r.loss(c.models, []string{"scifact"}, c.p); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("loss(%v): %v; want an error naming %q", c.models, err, c.want)
		}
	}
	if _, err := m.r.recall100("a", "scifact", confirmatory(1).run(10)); err == nil || !strings.Contains(err.Error(), "10 hits") {
		t.Errorf("recall100 at ef 10: %v; want an error on depth", err)
	}
	if _, err := m.r.recall100("a", "scifact", "exact"); err != nil {
		t.Errorf("recall100 of the exact run: %v", err)
	}
}

// TestReadingSets checks the rule on NQ as tables applies it: the models
// stored on each dataset are the models encoded there.
func TestReadingSets(t *testing.T) {
	models := []string{"m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"}
	setup := func(onNQ int) *results {
		m := newMem()
		for _, ds := range study.ConfirmatoryDatasets {
			for i, model := range models {
				if ds != "nq" || i < onNQ {
					m.cell(model, ds, []string{"q"}, map[string][]float64{"exact": {0}})
				}
			}
		}
		return m.r
	}
	cases := []struct {
		onNQ    int
		failed  []string
		tier1   bool
		nq      bool
		cmodels int
	}{
		{onNQ: 6, nq: true, cmodels: 6},
		{onNQ: 8, nq: true, cmodels: 8},
		{onNQ: 5, nq: false, cmodels: 8},
		{onNQ: 6, failed: []string{"m1"}, nq: false, cmodels: 7},
		{onNQ: 7, failed: []string{"m8"}, nq: true, cmodels: 7},
		{onNQ: 8, tier1: true, nq: false, cmodels: 8},
	}
	for _, c := range cases {
		s := setup(c.onNQ).readingSets(c.failed, c.tier1)
		if slices.Contains(s.cdatasets, "nq") != c.nq || len(s.cmodels) != c.cmodels {
			t.Errorf("%d on NQ, failed %v, tier 1 %v: models %v, datasets %v", c.onNQ, c.failed, c.tier1, s.cmodels, s.cdatasets)
		}
		if len(c.failed) > 0 && slices.Contains(s.models, c.failed[0]) {
			t.Errorf("failed model %s still read", c.failed[0])
		}
	}
}

func TestReadCellRejects(t *testing.T) {
	dir := t.TempDir()
	for _, c := range []struct{ name, body, want string }{
		{"header", "run\tqid\n", "header"},
		{"no exact", storeHeader + "\nm16-efc200-s1/ef40\tq1\t0.5\t-\t0\t1\t10\t-\t-\n", "no exact"},
		{"qids", storeHeader + "\nexact\tq1\t0.5\t0.5\t0\t0\t0\t0\t0\nexact\tq2\t0.5\t0.5\t0\t0\t0\t0\t0\n" +
			"rrf-exact\tq2\t0.5\t0.5\t0\t0\t0\t0\t0\nrrf-exact\tq1\t0.5\t0.5\t0\t0\t0\t0\t0\n", "qids of rrf-exact"},
	} {
		path := filepath.Join(dir, c.name+".tsv.gz")
		writeGz(t, path, c.body)
		if _, err := readCell(path); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v; want an error naming %q", c.name, err, c.want)
		}
	}
}

// TestTables runs collect and tables on synthetic measures of three models
// on the six small datasets and TREC-COVID: NQ is not stored, so it
// leaves the confirmatory set.
func TestTables(t *testing.T) {
	in, out := t.TempDir(), t.TempDir()
	for i, model := range []string{"bge-base-en-v1.5", "e5-base-v2", "gte-base"} {
		for _, ds := range []string{"scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "quora", "trec-covid"} {
			synth{queries: 8, loss: 0.002 * float64(i+1), m8m32: i == 0, anchor: i == 0}.write(t, filepath.Join(in, model, ds))
		}
	}
	if err := collect(in, out, maxFile); err != nil {
		t.Fatal(err)
	}
	doc := filepath.Join(t.TempDir(), "STUDY.md")
	start := "# Study\n\n<!-- confirmatory -->\nnot yet\n<!-- /confirmatory -->\n\n<!-- exploratory -->\nnot yet\n<!-- /exploratory -->\n"
	if err := os.WriteFile(doc, []byte(start), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := tables(out, doc, nil, false); err != nil {
		t.Fatal(err)
	}
	first := readFile(t, doc)
	for _, want := range []string{
		"Confirmatory datasets (6): scifact, nfcorpus, arguana, scidocs, fiqa, quora.",
		"| H1 |", "| H2 |", "| H3 |", "| H4 |", "| H5 |",
		"| recall 0.80, closest |", "Labels per dataset at recall 0.95", "| trec-covid |",
		"Normalize fusion", "| bge-base-en-v1.5 | quora |",
	} {
		if !strings.Contains(first, want) {
			t.Errorf("tables output lacks %q", want)
		}
	}
	if strings.Contains(first, "not yet") {
		t.Error("the markers' old content is kept")
	}
	if err := tables(out, doc, nil, false); err != nil {
		t.Fatal(err)
	}
	if again := readFile(t, doc); again != first {
		t.Error("a second run of tables writes something else")
	}
	if err := tables(out, doc, []string{"e5-base-v2", "gte-base"}, false); err == nil {
		t.Error("want an error with one confirmatory model left")
	}
}
