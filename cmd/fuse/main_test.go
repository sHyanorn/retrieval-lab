package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// file writes text to a file in dir and returns its path.
func file(t *testing.T, dir, name, text string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// fuse runs the command with args on the runs bm25 and dense and returns
// the RRF and Normalize runs it wrote.
func fuse(t *testing.T, bm25, dense string, args ...string) (rrf, norm string) {
	t.Helper()
	dir := t.TempDir()
	r, n := filepath.Join(dir, "rrf.run"), filepath.Join(dir, "norm.run")
	in, s, err := parse(flag.NewFlagSet("fuse", flag.ContinueOnError), append([]string{"-bm25", bm25, "-bge", dense, "-rrf", r, "-norm", n}, args...))
	if err != nil {
		t.Fatal(err)
	}
	if err := run(in, s); err != nil {
		t.Fatal(err)
	}
	a, err := os.ReadFile(r)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(n)
	if err != nil {
		t.Fatal(err)
	}
	return string(a), string(b)
}

// Without -study the outputs are those the command wrote before -study
// existed, byte for byte.
func TestDefaultsUnchanged(t *testing.T) {
	dir := t.TempDir()
	bm25 := file(t, dir, "bm25.run", "q1 Q0 c 1 3.000000 bm25\nq1 Q0 b 2 2.500000 bm25\nq1 Q0 a 3 2.500000 bm25\nq2 Q0 e 1 1.000000 bm25\n")
	dense := file(t, dir, "dense.run", "q1 Q0 a 1 0.900000 t\nq1 Q0 b 2 0.800000 t\nq1 Q0 d 3 0.800000 t\nq1 Q0 c 4 0.700000 t\nq2 Q0 e 1 0.600000 t\nq2 Q0 f 2 0.500000 t\n")
	for _, c := range []struct {
		args      []string
		rrf, norm string
	}{
		{nil,
			"q1 Q0 a 1 0.032266458495966696 rrf\nq1 Q0 b 2 0.03225806451612903 rrf\nq1 Q0 c 3 0.032018442622950824 rrf\nq1 Q0 d 4 0.015873015873015872 rrf\nq2 Q0 e 1 0.03278688524590164 rrf\nq2 Q0 f 2 0.016129032258064516 rrf\n",
			"q1 Q0 a 1 1 normalize\nq1 Q0 c 2 1 normalize\nq1 Q0 b 3 0.5000000000000002 normalize\nq1 Q0 d 4 0.5000000000000002 normalize\nq2 Q0 e 1 2 normalize\nq2 Q0 f 2 0 normalize\n"},
		{[]string{"-dense-depth", "2", "-dense-rank", "trec"},
			"q1 Q0 a 1 0.032266458495966696 rrf\nq1 Q0 c 2 0.01639344262295082 rrf\nq1 Q0 b 3 0.016129032258064516 rrf\nq1 Q0 d 4 0.016129032258064516 rrf\nq2 Q0 e 1 0.03278688524590164 rrf\nq2 Q0 f 2 0.016129032258064516 rrf\n",
			"q1 Q0 a 1 1 normalize\nq1 Q0 c 2 1 normalize\nq1 Q0 b 3 0 normalize\nq1 Q0 d 4 0 normalize\nq2 Q0 e 1 2 normalize\nq2 Q0 f 2 0 normalize\n"},
	} {
		if rrf, norm := fuse(t, bm25, dense, c.args...); rrf != c.rrf || norm != c.norm {
			t.Errorf("%v: wrote\n%s%s, want\n%s%s", c.args, rrf, norm, c.rrf, c.norm)
		}
	}
}

// -study sets the three settings of the study, refuses them given beside
// it, cuts the dense run at 100 and tags its outputs.
func TestStudy(t *testing.T) {
	_, s, err := parse(flag.NewFlagSet("fuse", flag.ContinueOnError), []string{"-study"})
	if want := (settings{depths: []int{1000, 100}, denseRank: "trec", suffix: "-study"}); err != nil || !reflect.DeepEqual(s, want) {
		t.Errorf("-study gives %+v, %v; want %+v", s, err, want)
	}
	for _, arg := range [][]string{{"-bm25-depth", "1000"}, {"-dense-depth", "100"}, {"-dense-rank", "trec"}} {
		_, _, err := parse(flag.NewFlagSet("fuse", flag.ContinueOnError), append([]string{"-study"}, arg...))
		if err == nil || !strings.Contains(err.Error(), arg[0]) {
			t.Errorf("-study with %s: error %v", arg[0], err)
		}
	}
	// 101 dense hits: the last one is fused by default and not by -study.
	dir := t.TempDir()
	var b strings.Builder
	for i := range 101 {
		fmt.Fprintf(&b, "q Q0 d%03d %d %f t\n", i, i+1, 1-float64(i)/1000)
	}
	bm25 := file(t, dir, "bm25.run", "q Q0 x 1 3.000000 bm25\n")
	dense := file(t, dir, "dense.run", b.String())
	rrf, norm := fuse(t, bm25, dense, "-study")
	if strings.Contains(rrf, " d100 ") || strings.Contains(norm, " d100 ") {
		t.Error("-study fused the dense run's 101st hit")
	}
	if def, _ := fuse(t, bm25, dense); !strings.Contains(def, " d100 ") {
		t.Error("the default did not fuse the dense run's 101st hit")
	}
	for _, out := range []struct{ text, tag string }{{rrf, "rrf-study"}, {norm, "normalize-study"}} {
		for _, line := range strings.Split(strings.TrimSuffix(out.text, "\n"), "\n") {
			if f := strings.Fields(line); f[5] != out.tag {
				t.Fatalf("-study wrote %q, want the tag %s", line, out.tag)
			}
		}
	}
	// Apart from the tags, -study writes what the three flags write.
	r3, n3 := fuse(t, bm25, dense, "-dense-depth", "100", "-dense-rank", "trec")
	if strings.ReplaceAll(rrf, " rrf-study\n", " rrf\n") != r3 || strings.ReplaceAll(norm, " normalize-study\n", " normalize\n") != n3 {
		t.Error("-study and -bm25-depth 1000 -dense-depth 100 -dense-rank trec fuse differently")
	}
}

// The study's two dense runs of the same documents, an exact run with ties
// by document ID ascending and an HNSW run as trec_eval ranks it, fuse to
// the same files with -dense-rank trec; by their lines, to different RRF
// files and the same Normalize file, which reads scores, not ranks.
func TestDenseRankTrec(t *testing.T) {
	dir := t.TempDir()
	bm25 := file(t, dir, "bm25.run", "q Q0 c 1 3.000000 bm25\n")
	exact := file(t, dir, "exact.run", "q Q0 a 1 1.000000 t\nq Q0 z 2 1.000000 t\nq Q0 c 3 0.500000 t\n")
	hnsw := file(t, dir, "hnsw.run", "q Q0 z 1 1.000000 t\nq Q0 a 2 1.000000 t\nq Q0 c 3 0.500000 t\n")
	for _, args := range [][]string{{"-dense-depth", "100", "-dense-rank", "trec"}, {"-study"}} {
		er, en := fuse(t, bm25, exact, args...)
		ar, an := fuse(t, bm25, hnsw, args...)
		if er != ar || en != an {
			t.Errorf("%v: the exact run fuses to\n%s%s, the HNSW run to\n%s%s", args, er, en, ar, an)
		}
	}
	er, en := fuse(t, bm25, exact, "-dense-depth", "100")
	ar, an := fuse(t, bm25, hnsw, "-dense-depth", "100")
	if er == ar {
		t.Errorf("-dense-rank line: both runs fuse by RRF to\n%s", er)
	}
	if en != an {
		t.Errorf("-dense-rank line: Normalize gives\n%s and\n%s", en, an)
	}
	in, s, err := parse(flag.NewFlagSet("fuse", flag.ContinueOnError), []string{"-bm25", bm25, "-bge", exact, "-rrf", filepath.Join(dir, "r"), "-norm", filepath.Join(dir, "n"), "-dense-rank", "rank"})
	if err != nil {
		t.Fatal(err)
	}
	if err := run(in, s); err == nil {
		t.Error("an unknown -dense-rank was accepted")
	}
}
