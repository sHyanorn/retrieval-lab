// Command embed encodes a BEIR corpus with BGE-base-en-v1.5 and writes the
// vectors as a faiss flat index directory (index and docid), the layout of
// Pyserini's prebuilt indexes. It prints the time taken as JSON.
//
// While it runs, the vectors are appended to vectors.part in the output
// directory every chunk documents; a run that finds that file continues
// after the vectors it holds.
package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
	"github.com/sHyanorn/retrieval-lab/internal/dense"
)

// chunk is the number of documents encoded between two appends to the
// partial file, the most an interruption loses.
const chunk = 1000

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
	if err := os.MkdirAll(out, 0o755); err != nil {
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
	part := filepath.Join(out, "vectors.part")
	start, encoded := time.Now(), 0
	vecs, resumed, err := encodeResumable(part, texts, func(t []string) ([]float32, error) {
		v, err := e.EncodeAll(t, workers, nil)
		encoded += len(t)
		fmt.Fprintf(os.Stderr, "embed: %d documents in this run, %.1f per second\n", encoded, float64(encoded)/time.Since(start).Seconds())
		return v, err
	})
	if err != nil {
		return err
	}
	secs := time.Since(start).Seconds()
	if err := dense.WriteFaissFlat(out, ids, vecs, dense.Dim); err != nil {
		return err
	}
	if err := os.Remove(part); err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{
		"docs": len(ids), "resumed_after": resumed, "seconds": int(secs),
		"docs_per_second": float64(len(ids)-resumed) / secs, "workers": workers,
	})
}

// encodeResumable returns the vectors of texts. It reads the whole vectors
// the file part holds as those of the first texts, encodes the rest chunk by
// chunk and writes each chunk after them, over a vector cut short. It also
// returns the number of vectors it read.
func encodeResumable(part string, texts []string, encode func([]string) ([]float32, error)) (vecs []float32, resumed int, err error) {
	f, err := os.OpenFile(part, os.O_RDWR|os.O_CREATE, 0o644) //nolint:gosec // data, not secrets
	if err != nil {
		return nil, 0, err
	}
	defer func() {
		if cerr := f.Close(); err == nil {
			err = cerr
		}
	}()
	raw, err := io.ReadAll(f)
	if err != nil {
		return nil, 0, err
	}
	const size = 4 * dense.Dim
	resumed = len(raw) / size
	if resumed > len(texts) {
		return nil, 0, fmt.Errorf("%s holds %d vectors for %d documents", part, resumed, len(texts))
	}
	if _, err := f.Seek(int64(resumed*size), io.SeekStart); err != nil {
		return nil, 0, err
	}
	vecs = make([]float32, len(texts)*dense.Dim)
	if err := binary.Read(bytes.NewReader(raw[:resumed*size]), binary.LittleEndian, vecs[:resumed*dense.Dim]); err != nil {
		return nil, 0, err
	}
	for i := resumed; i < len(texts); i += chunk {
		v, err := encode(texts[i:min(i+chunk, len(texts))])
		if err != nil {
			return nil, resumed, err
		}
		copy(vecs[i*dense.Dim:], v)
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			return nil, resumed, err
		}
		if err := f.Sync(); err != nil {
			return nil, resumed, err
		}
	}
	return vecs, resumed, nil
}
