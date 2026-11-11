// Command embed encodes a BEIR corpus with BGE-base-en-v1.5 and writes the
// vectors as a faiss flat index directory (index and docid), the layout of
// Pyserini's prebuilt indexes. It prints the time taken as JSON.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/dense"
)

func main() {
	data := flag.String("data", "", "BEIR dataset directory")
	out := flag.String("out", "", "output directory")
	model := flag.String("model", "", "directory holding bge-base-en-v1.5-optimized.onnx and tokenizer.json")
	ortLib := flag.String("ort", "", "path to libonnxruntime.so")
	workers := flag.Int("workers", runtime.GOMAXPROCS(0), "concurrent encoder calls, one thread each")
	flag.Parse()
	if err := run(*data, *out, *model, *ortLib, *workers); err != nil {
		fmt.Fprintln(os.Stderr, "embed:", err)
		os.Exit(1)
	}
}

func run(data, out, model, ortLib string, workers int) error {
	var ids, texts []string
	if err := beir.Corpus(data, func(d beir.Doc) error {
		ids = append(ids, d.ID)
		texts = append(texts, dense.DocText(d))
		return nil
	}); err != nil {
		return err
	}
	if err := dense.Init(ortLib); err != nil {
		return err
	}
	e, err := dense.NewEncoder(filepath.Join(model, "bge-base-en-v1.5-optimized.onnx"), filepath.Join(model, "tokenizer.json"), 512)
	if err != nil {
		return err
	}
	defer e.Close() //nolint:errcheck // the vectors are written by then
	start := time.Now()
	vecs, err := e.EncodeAll(texts, workers, func(done int) {
		if done%1000 == 0 {
			fmt.Fprintf(os.Stderr, "embed: %d of %d documents, %.1f per second\n", done, len(texts), float64(done)/time.Since(start).Seconds())
		}
	})
	if err != nil {
		return err
	}
	secs := time.Since(start).Seconds()
	if err := dense.WriteFaissFlat(out, ids, vecs, dense.Dim); err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{
		"docs": len(ids), "seconds": int(secs), "docs_per_second": float64(len(ids)) / secs, "workers": workers,
	})
}
