package main

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/dense"
)

// fakeEncode gives text i a vector whose first component is i and counts
// the texts it encodes; after failAfter calls it fails.
func fakeEncode(count *int, failAfter int) func([]string) ([]float32, error) {
	calls := 0
	return func(texts []string) ([]float32, error) {
		if calls++; calls > failAfter {
			return nil, errors.New("interrupted")
		}
		*count += len(texts)
		v := make([]float32, len(texts)*dense.Dim)
		for j, t := range texts {
			i, err := strconv.Atoi(t)
			if err != nil {
				return nil, err
			}
			v[j*dense.Dim] = float32(i)
		}
		return v, nil
	}
}

func TestEncodeResumable(t *testing.T) {
	texts := make([]string, 2*chunk+chunk/2)
	for i := range texts {
		texts[i] = strconv.Itoa(i)
	}
	want, _, err := encodeResumable(filepath.Join(t.TempDir(), "part"), texts, fakeEncode(new(int), len(texts)))
	if err != nil {
		t.Fatal(err)
	}

	part := filepath.Join(t.TempDir(), "part")
	if _, _, err := encodeResumable(part, texts, fakeEncode(new(int), 1)); err == nil {
		t.Fatal("no error from an interrupted encoding")
	}
	// A vector cut short by the interruption.
	f, err := os.OpenFile(part, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(make([]byte, 10)); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	encoded := 0
	got, resumed, err := encodeResumable(part, texts, fakeEncode(&encoded, len(texts)))
	if err != nil {
		t.Fatal(err)
	}
	if resumed != chunk || encoded != len(texts)-chunk {
		t.Errorf("resumed after %d and encoded %d texts, want %d and %d", resumed, encoded, chunk, len(texts)-chunk)
	}
	if !slices.Equal(got, want) {
		t.Error("resumed vectors differ from an uninterrupted encoding")
	}
	// The file now holds every vector, in order.
	got, _, err = encodeResumable(part, texts, fakeEncode(new(int), 0))
	if err != nil || !slices.Equal(got, want) {
		t.Errorf("vectors read back from the partial file differ (err %v)", err)
	}
}
