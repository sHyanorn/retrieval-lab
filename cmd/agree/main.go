// Command agree compares two sets of vectors, each a faiss flat index
// directory (index and docid), by document ID. For every document of -ref it
// takes the cosine to the vector of the same ID in -vectors and prints the
// summary (dense.Agreement) as JSON. It exits 1 if a cosine is below 0.9999,
// D012's threshold, which checks 1 and 2 of docs/PREREGISTRATION.md, Section 5
// apply, and 2 on an error. -vectors may hold documents -ref does not, unless
// -all is given.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/sHyanorn/retrieval-lab/internal/dense"
)

func main() {
	ref := flag.String("ref", "", "reference vectors (faiss flat directory)")
	vecs := flag.String("vectors", "", "vectors to check (faiss flat directory)")
	all := flag.Bool("all", false, "require both to hold the same documents")
	flag.Parse()
	g, err := agree(*ref, *vecs, *all)
	if err == nil {
		err = json.NewEncoder(os.Stdout).Encode(g)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "agree:", err)
		os.Exit(2)
	}
	if g.Below9999 > 0 {
		os.Exit(1)
	}
}

// agree compares the vectors of refDir with those of the same IDs in dir.
func agree(refDir, dir string, all bool) (dense.Agreement, error) {
	refIDs, ref, dim, err := dense.ReadFaissFlat(refDir)
	if err != nil {
		return dense.Agreement{}, err
	}
	ids, vecs, d, err := dense.ReadFaissFlat(dir)
	if err != nil {
		return dense.Agreement{}, err
	}
	if d != dim {
		return dense.Agreement{}, fmt.Errorf("dimension %d against %d", d, dim)
	}
	if all && len(ids) != len(refIDs) {
		return dense.Agreement{}, fmt.Errorf("%d documents against %d", len(ids), len(refIDs))
	}
	row := make(map[string]int, len(ids))
	for i, id := range ids {
		row[id] = i
	}
	if len(row) != len(ids) {
		return dense.Agreement{}, fmt.Errorf("%s holds a document ID twice", dir)
	}
	sub := make([]float32, len(refIDs)*dim)
	for i, id := range refIDs {
		j, ok := row[id]
		if !ok {
			return dense.Agreement{}, fmt.Errorf("document %s is missing from %s", id, dir)
		}
		copy(sub[i*dim:(i+1)*dim], vecs[j*dim:(j+1)*dim])
	}
	return dense.Compare(refIDs, ref, refIDs, sub, dim)
}
