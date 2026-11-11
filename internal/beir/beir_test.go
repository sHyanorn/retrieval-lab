package beir

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

func TestReadQrelsAcceptsCRLF(t *testing.T) {
	in := "query-id\tcorpus-id\tscore\r\nq1\td1\t2\r\nq1\td2\t-1\r\nq2\td1\t0\r\n"
	got, err := readQrels(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	want := eval.Qrels{"q1": {"d1": 2, "d2": -1}, "q2": {"d1": 0}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestReadQrelsRejectsDuplicatesAndMissingHeader(t *testing.T) {
	for _, in := range []string{
		"query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td1\t2\n",
		"q1\td1\t1\n",
	} {
		if _, err := readQrels(strings.NewReader(in)); err == nil {
			t.Errorf("want an error for %q", in)
		}
	}
}

func TestCorpusAndQueries(t *testing.T) {
	dir := t.TempDir()
	write := func(name, s string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(s), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("corpus.jsonl", `{"_id": "d1", "title": "T", "text": "a ’b", "metadata": {"x": 1}}`+"\n"+
		`{"_id": "d2", "title": "", "text": "c"}`+"\n")
	write("queries.jsonl", `{"_id": "q1", "text": "what?", "metadata": {}}`+"\n")

	var docs []Doc
	if err := Corpus(dir, func(d Doc) error { docs = append(docs, d); return nil }); err != nil {
		t.Fatal(err)
	}
	want := []Doc{{ID: "d1", Title: "T", Text: "a ’b"}, {ID: "d2", Text: "c"}}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("corpus: got %v, want %v", docs, want)
	}
	q, err := Queries(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(q, map[string]string{"q1": "what?"}) {
		t.Fatalf("queries: got %v", q)
	}
}
