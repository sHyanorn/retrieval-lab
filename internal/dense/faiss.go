package dense

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// ReadFaissFlat reads a Pyserini faiss flat index directory: "index", a
// faiss IndexFlatIP (faiss/impl/index_write.cpp), and "docid", one document
// ID per line for each row. It is used only to compare vectors.
func ReadFaissFlat(dir string) (ids []string, vecs []float32, dim int, err error) {
	raw, err := os.ReadFile(filepath.Join(dir, "index"))
	if err != nil {
		return nil, nil, 0, err
	}
	// "IxFI", int32 d, int64 ntotal, two int64 placeholders, uint8
	// is_trained, int32 metric, uint64 count, then count float32 values.
	const header = 45
	if len(raw) < header || string(raw[:4]) != "IxFI" {
		return nil, nil, 0, fmt.Errorf("%s: not a faiss IndexFlatIP", dir)
	}
	dim = int(int32(binary.LittleEndian.Uint32(raw[4:])))
	n := int(binary.LittleEndian.Uint64(raw[8:]))
	count := int(binary.LittleEndian.Uint64(raw[37:]))
	if metric := binary.LittleEndian.Uint32(raw[33:]); metric != 0 || count != n*dim || len(raw) != header+4*count {
		return nil, nil, 0, fmt.Errorf("%s: unexpected faiss header (metric %d, %d values for %d×%d, %d bytes)", dir, metric, count, n, dim, len(raw))
	}
	vecs = make([]float32, count)
	for i := range vecs {
		vecs[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[header+4*i:]))
	}
	f, err := os.Open(filepath.Join(dir, "docid"))
	if err != nil {
		return nil, nil, 0, err
	}
	defer f.Close() //nolint:errcheck // read-only
	ids, err = readLines(f)
	if err == nil && len(ids) != n {
		err = fmt.Errorf("%s: %d document IDs for %d vectors", dir, len(ids), n)
	}
	return ids, vecs, dim, err
}

func readLines(r io.Reader) ([]string, error) {
	var out []string
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		out = append(out, strings.TrimRight(sc.Text(), " \t\r"))
	}
	return out, sc.Err()
}

// WriteFaissFlat writes vectors in the layout ReadFaissFlat reads, so that
// this project's vectors and Pyserini's can be read and compared alike.
func WriteFaissFlat(dir string, ids []string, vecs []float32, dim int) error {
	if len(vecs) != len(ids)*dim {
		return fmt.Errorf("%d values for %d IDs of dimension %d", len(vecs), len(ids), dim)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	buf := make([]byte, 45+4*len(vecs))
	copy(buf, "IxFI")
	binary.LittleEndian.PutUint32(buf[4:], uint32(dim))
	binary.LittleEndian.PutUint64(buf[8:], uint64(len(ids)))
	binary.LittleEndian.PutUint64(buf[16:], 1<<20)
	binary.LittleEndian.PutUint64(buf[24:], 1<<20)
	buf[32] = 1
	binary.LittleEndian.PutUint64(buf[37:], uint64(len(vecs)))
	for i, v := range vecs {
		binary.LittleEndian.PutUint32(buf[45+4*i:], math.Float32bits(v))
	}
	if err := os.WriteFile(filepath.Join(dir, "index"), buf, 0o644); err != nil { //nolint:gosec // data, not secrets
		return err
	}
	return os.WriteFile(filepath.Join(dir, "docid"), []byte(strings.Join(ids, "\n")+"\n"), 0o644) //nolint:gosec // data, not secrets
}
