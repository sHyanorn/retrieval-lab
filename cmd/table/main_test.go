package main

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

func TestAgreementFollowsSpec(t *testing.T) {
	for _, c := range []struct {
		ours, anchor float64
		want         string
	}{
		{0.67894, 0.6789, "all four decimals"},
		{0.6794, 0.6789, "three printed decimals"},
		{0.6739, 0.6789, "within 0.005"},
		{0.6738, 0.6789, "outside 0.005"},
	} {
		if got := agreement(delta(c.ours, c.anchor)); got != c.want {
			t.Errorf("%v against %v: %q, want %q", c.ours, c.anchor, got, c.want)
		}
	}
}

func TestSpliceReplacesOnlyTheMarkedBlock(t *testing.T) {
	got, err := splice("a\n<!-- x -->\nold\n<!-- /x -->\nb\n", "x", "new\n")
	if want := "a\n<!-- x -->\nnew\n<!-- /x -->\nb\n"; err != nil || got != want {
		t.Fatalf("splice = %q, %v; want %q", got, err, want)
	}
	if _, err := splice("no markers", "x", ""); err == nil {
		t.Fatal("want an error without markers")
	}
}

func TestAnchorReadersRejectTheOtherFile(t *testing.T) {
	if _, err := readDenseAnchors("../../results/anchors.tsv"); err == nil {
		t.Error("readDenseAnchors accepted the BM25 anchors")
	}
	if _, err := readAnchors("../../results/anchors-bge.tsv"); err == nil {
		t.Error("readAnchors accepted the BGE anchors")
	}
	if _, err := readDenseAnchors("../../results/anchors-bge.tsv"); err != nil {
		t.Error(err)
	}
}

func TestMeanOfTellsMissingQrelsFromMissingRun(t *testing.T) {
	dir := t.TempDir()
	run := filepath.Join(dir, "d.run")
	if err := os.WriteFile(run, []byte("q1 Q0 d1 1 0.5 t\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := meanOf(filepath.Join(dir, "nodata"), run); err == nil || errors.Is(err, fs.ErrNotExist) {
		t.Errorf("run present, qrels missing: %v, want an error that is not fs.ErrNotExist", err)
	}
}

func TestDenseTablesReportARunOutsideTheTolerance(t *testing.T) {
	dir := t.TempDir()
	anchors := filepath.Join(dir, "anchors.tsv")
	const tsv = "dataset\tflat_ndcg_cut_10\tflat_recall_100\tfaiss_ndcg_cut_10\tfaiss_recall_100\nd\t0.6000\t0.9000\t0.6000\t0.9000\n"
	if err := os.WriteFile(anchors, []byte(tsv), 0o600); err != nil {
		t.Fatal(err)
	}
	mean := func(string, string) (eval.Scores, error) { return eval.Scores{NDCG10: 0.5, Recall100: 0.9}, nil }
	const doc = "<!-- bge-flat -->\n<!-- /bge-flat -->\n<!-- bge-vectors -->\n<!-- /bge-vectors -->\n"
	_, short, err := denseTables(doc, anchors, dir, dir, dir, mean)
	if err != nil {
		t.Fatal(err)
	}
	if len(short) != 1 || !strings.HasPrefix(short[0], "d: nDCG@10 0.5000") {
		t.Errorf("short = %q, want one line for d at nDCG@10 0.5000", short)
	}
}

func TestCheckLeavesTheDocumentAlone(t *testing.T) {
	dir := t.TempDir()
	doc := filepath.Join(dir, "doc.md")
	const before = "<!-- bge-flat -->\nkept\n<!-- /bge-flat -->\n<!-- bge-vectors -->\nkept\n<!-- /bge-vectors -->\n"
	if err := os.WriteFile(doc, []byte(before), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := run("../../results/anchors-bge.tsv", dir, dir, doc, "", false, false, true, false, false, false, true, dir, dir); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(doc)
	if err != nil || string(after) != before {
		t.Fatalf("-check rewrote the document (%v):\n%s", err, after)
	}
}

func TestVariantsNDCGTableMatchesTheResults(t *testing.T) {
	doc, err := os.ReadFile("../../docs/BENCHMARKS.md")
	if err != nil {
		t.Fatal(err)
	}
	results, err := os.ReadFile("../../results/bleve-variants.md")
	if err != nil {
		t.Fatal(err)
	}
	const start, end = "<!-- variants-ndcg -->\n", "<!-- /variants-ndcg -->"
	r := string(results)
	i, j := strings.Index(r, start), strings.Index(r, end)
	if i < 0 || j < i {
		t.Fatal("results/bleve-variants.md holds no nDCG@10 table")
	}
	got, err := splice(string(doc), "variants-ndcg", r[i+len(start):j])
	if err != nil {
		t.Fatal(err)
	}
	if got != string(doc) {
		t.Fatal("the nDCG@10 table of docs/BENCHMARKS.md differs from that of results/bleve-variants.md; run make variants")
	}
}

func TestNDCGChangeRoundsTheDifference(t *testing.T) {
	// 0.402553 − 0.406231 = −0.003678; rounding each first would give
	// 0.4026 − 0.4062 = −0.0036.
	if got := ndcgChange(0.402553, 0.406231); got != "-0.0037" {
		t.Fatalf("ndcgChange = %s, want -0.0037", got)
	}
}

func TestAnchorBuildTablesMatchTheDocument(t *testing.T) {
	doc, err := os.ReadFile("../../docs/BENCHMARKS.md")
	if err != nil {
		t.Fatal(err)
	}
	got, err := anchorBuildTables(string(doc), "../../results/anchors-hnsw.tsv", "../../study/results")
	if err != nil {
		t.Fatal(err)
	}
	if got != string(doc) {
		t.Fatal("the anchor build's tables in docs/BENCHMARKS.md differ from those of study/results; run make anchor-build-table")
	}
}

func TestDifferingSeparatesTiesFromMoves(t *testing.T) {
	h := func(doc string, score float64) eval.Hit { return eval.Hit{DocID: doc, Score: score} }
	a := eval.Run{
		"q1": {h("x", 0.5), h("y", 0.499999), h("z", 0.3)},
		"q2": {h("x", 0.9), h("y", 0.4), h("z", 0.1)},
		"q3": {h("x", 0.2), h("y", 0.1)},
	}
	b := eval.Run{
		// x and y tie in a and are one step of the fourth decimal apart in
		// b, as two scores on either side of a rounding boundary can be.
		"q1": {h("y", 0.5), h("x", 0.4999), h("z", 0.3)},
		// x and y swap places, 0.5 apart.
		"q2": {h("y", 0.9), h("x", 0.4), h("z", 0.1)},
		// The last place holds w, which a does not hold; w scores as y does.
		"q3": {h("x", 0.2), h("w", 0.1)},
	}
	// 150 documents tie at 0.2579 and come in the opposite order: the tie
	// adjustment lowers the last by 0.000149, yet all are ties.
	for k := range 150 {
		s := 0.2579 - 1e-6*float64(k)
		a["q4"] = append(a["q4"], h(fmt.Sprintf("d%03d", k), s))
		b["q4"] = append(b["q4"], h(fmt.Sprintf("d%03d", 149-k), s))
	}
	if n, apart := differing(a, b); n != 155 || apart != 2 {
		t.Fatalf("differing = %d, %d apart; want 155, 2 apart", n, apart)
	}
}

func TestVariantsWriteTheirNDCGTableIntoTheDocument(t *testing.T) {
	tables := map[string]string{"variants-ndcg": "ndcg\n", "variants-lines": "lines\n", "variants-bleve": "bleve\n", "variants-stats": "stats\n"}
	var results, want string
	for _, name := range []string{"variants-ndcg", "variants-lines", "variants-bleve", "variants-stats"} {
		results += "<!-- " + name + " -->\nold\n<!-- /" + name + " -->\n"
		want += "<!-- " + name + " -->\n" + tables[name] + "<!-- /" + name + " -->\n"
	}
	// The document holds the nDCG@10 table alone.
	doc := "a\n<!-- variants-ndcg -->\nold\n<!-- /variants-ndcg -->\nb\n"
	r, d, err := spliceVariants(results, doc, tables)
	if err != nil {
		t.Fatal(err)
	}
	if r != want {
		t.Errorf("results = %q, want %q", r, want)
	}
	if want := "a\n<!-- variants-ndcg -->\nndcg\n<!-- /variants-ndcg -->\nb\n"; d != want {
		t.Errorf("document = %q, want %q", d, want)
	}
	if _, _, err := spliceVariants(results, "no markers", tables); err == nil {
		t.Error("want an error for a document without the nDCG@10 markers")
	}
}
