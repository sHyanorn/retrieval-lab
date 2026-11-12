package dense

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
)

// TestDocTextAsEncodeScript checks DocText against the cases that
// encode/test_encode.py checks the encoding script's doc_text against.
func TestDocTextAsEncodeScript(t *testing.T) {
	raw, err := os.ReadFile("../../encode/testdata/doctext.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct{ Title, Text, Want string }
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("no cases")
	}
	for _, c := range cases {
		if got := DocText(beir.Doc{Title: c.Title, Text: c.Text}); got != c.Want {
			t.Errorf("DocText(%q, %q) = %q, want %q", c.Title, c.Text, got, c.Want)
		}
	}
}

// TestReadsEncodeScriptLayout reads vectors that encode/encode.py wrote and
// writes them back byte for byte.
func TestReadsEncodeScriptLayout(t *testing.T) {
	src := filepath.Join("testdata", "python-flat")
	ids, vecs, dim, err := ReadFaissFlat(src)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0.5, -0.25, 1e-30, 0, 3.4e38, 0.1}
	if dim != 3 || !slices.Equal(ids, []string{"doc-1", "doc 2"}) || !slices.Equal(vecs, want) {
		t.Fatalf("read %v, %v, dim %d", ids, vecs, dim)
	}
	out := filepath.Join(t.TempDir(), "v")
	if err := WriteFaissFlat(out, ids, vecs, dim); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"index", "docid"} {
		a, errA := os.ReadFile(filepath.Join(src, f))
		b, errB := os.ReadFile(filepath.Join(out, f))
		if errA != nil || errB != nil || !bytes.Equal(a, b) {
			t.Errorf("%s differs from the script's (%v, %v)", f, errA, errB)
		}
	}
}
