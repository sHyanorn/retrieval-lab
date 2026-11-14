package main

import (
	"os"
	"path/filepath"
	"testing"
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

func TestCheckLeavesTheDocumentAlone(t *testing.T) {
	dir := t.TempDir()
	doc := filepath.Join(dir, "doc.md")
	const before = "<!-- bge-flat -->\nkept\n<!-- /bge-flat -->\n<!-- bge-vectors -->\nkept\n<!-- /bge-vectors -->\n"
	if err := os.WriteFile(doc, []byte(before), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := run("../../results/anchors-bge.tsv", dir, dir, doc, false, true, false, false, false, true, dir, dir); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(doc)
	if err != nil || string(after) != before {
		t.Fatalf("-check rewrote the document (%v):\n%s", err, after)
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
