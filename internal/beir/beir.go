// Package beir reads a BEIR dataset directory: corpus.jsonl, queries.jsonl
// and qrels/<split>.tsv.
package beir

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/eval"
)

// Doc is one corpus document.
type Doc struct {
	ID    string `json:"_id"`
	Title string `json:"title"`
	Text  string `json:"text"`
}

// Corpus calls fn for each document of dir/corpus.jsonl in file order.
func Corpus(dir string, fn func(Doc) error) error {
	f, err := os.Open(filepath.Join(dir, "corpus.jsonl"))
	if err != nil {
		return err
	}
	defer f.Close() //nolint:errcheck // read-only
	dec := json.NewDecoder(bufio.NewReader(f))
	for {
		var d Doc
		if err := dec.Decode(&d); errors.Is(err, io.EOF) {
			return nil
		} else if err != nil {
			return fmt.Errorf("corpus.jsonl: %w", err)
		}
		if err := fn(d); err != nil {
			return err
		}
	}
}

// Queries reads dir/queries.jsonl into a map from query ID to text.
func Queries(dir string) (map[string]string, error) {
	f, err := os.Open(filepath.Join(dir, "queries.jsonl"))
	if err != nil {
		return nil, err
	}
	defer f.Close() //nolint:errcheck // read-only
	q := map[string]string{}
	dec := json.NewDecoder(bufio.NewReader(f))
	for {
		var d struct {
			ID   string `json:"_id"`
			Text string `json:"text"`
		}
		if err := dec.Decode(&d); errors.Is(err, io.EOF) {
			return q, nil
		} else if err != nil {
			return nil, fmt.Errorf("queries.jsonl: %w", err)
		}
		q[d.ID] = d.Text
	}
}

// Qrels reads dir/qrels/<split>.tsv: a header line "query-id corpus-id
// score", then tab-separated lines. BEIR ships these files with CRLF line
// endings.
func Qrels(dir, split string) (eval.Qrels, error) {
	f, err := os.Open(filepath.Join(dir, "qrels", split+".tsv"))
	if err != nil {
		return nil, err
	}
	defer f.Close() //nolint:errcheck // read-only
	return readQrels(f)
}

func readQrels(r io.Reader) (eval.Qrels, error) {
	sc := bufio.NewScanner(r)
	line := func() string { return strings.TrimSuffix(sc.Text(), "\r") }
	if !sc.Scan() || line() != "query-id\tcorpus-id\tscore" {
		return nil, fmt.Errorf("qrels: missing header line, got %q", line())
	}
	q := eval.Qrels{}
	for n := 2; sc.Scan(); n++ {
		f := strings.Split(line(), "\t")
		if len(f) != 3 {
			return nil, fmt.Errorf("qrels line %d: want 3 fields, got %d", n, len(f))
		}
		grade, err := strconv.Atoi(f[2])
		if err != nil {
			return nil, fmt.Errorf("qrels line %d: %w", n, err)
		}
		if q[f[0]] == nil {
			q[f[0]] = map[string]int{}
		}
		if _, dup := q[f[0]][f[1]]; dup {
			return nil, fmt.Errorf("qrels line %d: document %s judged twice for query %s", n, f[1], f[0])
		}
		q[f[0]][f[1]] = grade
	}
	return q, sc.Err()
}
