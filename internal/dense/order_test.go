package dense

import (
	"bytes"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/sHyanorn/retrieval-lab/internal/eval"
	"github.com/sHyanorn/retrieval-lab/internal/hnsw"
)

// corpus returns n random unit vectors of dimension dim with IDs d000 up.
func corpus(n, dim int) *Index {
	r := rand.New(rand.NewPCG(3, 4))
	x := &Index{}
	for i := range n {
		v := make([]float32, dim)
		var norm float64
		for j := range v {
			v[j] = float32(r.NormFloat64())
			norm += float64(v[j]) * float64(v[j])
		}
		for j := range v {
			v[j] /= float32(math.Sqrt(norm))
		}
		x.IDs = append(x.IDs, fmt.Sprintf("d%03d", i))
		x.Vecs = append(x.Vecs, v...)
	}
	return x
}

func graphBytes(t *testing.T, g *hnsw.Index) []byte {
	t.Helper()
	var b bytes.Buffer
	if _, err := g.WriteTo(&b); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// inOrder reports whether row k of x holds document p[k] of the corpus c.
func inOrder(x, c *Index, p []int, dim int) bool {
	for k, i := range p {
		if x.IDs[k] != c.IDs[i] || !slices.Equal(x.Vecs[k*dim:(k+1)*dim], c.Vecs[i*dim:(i+1)*dim]) {
			return false
		}
	}
	return true
}

func TestSeedZeroIsTheM3Build(t *testing.T) {
	const n, dim = 300, 8
	x, c := corpus(n, dim), corpus(n, dim)
	g, err := x.BuildGraph(16, 200, 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := hnsw.Build(c.Vecs, dim, GraphConfig)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(graphBytes(t, g), graphBytes(t, want)) {
		t.Error("seed 0 at M 16, efConstruction 200 is not GraphConfig's graph")
	}
	if !slices.Equal(x.IDs, c.IDs) || !slices.Equal(x.Vecs, c.Vecs) {
		t.Error("seed 0 moved the rows")
	}
}

func TestArrangeInsertsDocumentPK(t *testing.T) {
	const n, dim = 200, 4
	c := corpus(n, dim)
	x := corpus(n, dim)
	for _, seed := range []uint64{3, 1, 2, 0, 4} {
		// The registered draw, written out here rather than called.
		p := make([]int, n)
		for i := range p {
			p[i] = i
		}
		if seed > 0 {
			p = rand.New(rand.NewPCG(20261111, seed)).Perm(n)
		}
		x.Arrange(seed)
		if !inOrder(x, c, p, dim) {
			t.Errorf("seed %d: row k does not hold document p[k]", seed)
		}
		if seed == 0 {
			continue
		}
		broken := corpus(n, dim)
		broken.arrange(seed, true)
		if inOrder(broken, c, p, dim) {
			t.Errorf("seed %d: the order p⁻¹ passed for p", seed)
		}
	}
}

func TestBuildGraphOfSeed(t *testing.T) {
	const n, dim = 300, 8
	c := corpus(n, dim)
	for _, seed := range []uint64{1, 2} {
		p := rand.New(rand.NewPCG(20261111, seed)).Perm(n)
		data := make([]float32, 0, n*dim)
		for _, i := range p {
			data = append(data, c.Vecs[i*dim:(i+1)*dim]...)
		}
		want, err := hnsw.Build(data, dim, hnsw.Config{M: 8, EfConstruction: 50, Seed: seed + 1, Workers: 1})
		if err != nil {
			t.Fatal(err)
		}
		x := corpus(n, dim)
		x.Arrange(seed + 5) // a later build starts from the corpus order again
		g, err := x.BuildGraph(8, 50, seed)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(graphBytes(t, g), graphBytes(t, want)) {
			t.Errorf("seed %d: graph differs from the build over the rows in order p", seed)
		}
		broken := corpus(n, dim)
		broken.arrange(seed, true)
		bg, err := hnsw.Build(broken.Vecs, dim, Graph(8, 50, seed))
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Equal(graphBytes(t, bg), graphBytes(t, want)) {
			t.Errorf("seed %d: the build in order p⁻¹ matched", seed)
		}
	}
}

func TestSearchAfterArrange(t *testing.T) {
	const n, dim = 300, 8
	x := corpus(n, dim)
	q := slices.Clone(x.Vecs[17*dim : 18*dim]) // d017's own vector
	exact := x.Search(q, n)
	g, err := x.BuildGraph(16, 200, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got := x.Search(q, n); !slices.Equal(got, exact) {
		t.Error("exact search changed with the order of the rows")
	}
	if x.IDs[17] == "d017" {
		t.Fatal("the test needs d017 moved")
	}
	// With ef = n the graph returns the exact top k; the skip must take out
	// d017 by its ID, wherever its node is.
	want := slices.DeleteFunc(slices.Clone(exact), func(h eval.Hit) bool { return h.DocID == "d017" })[:10]
	if got := x.SearchGraph(g, q, 10, n, "d017"); !slices.Equal(got, want) {
		t.Errorf("SearchGraph with skip = %v, want %v", got, want)
	}
	if got := x.SearchGraph(g, q, 10, n, ""); !slices.Equal(got, exact[:10]) || got[0].DocID != "d017" {
		t.Errorf("SearchGraph = %v, want %v", got, exact[:10])
	}
}
