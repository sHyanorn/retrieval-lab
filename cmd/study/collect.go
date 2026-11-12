package main

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"fmt"
	"maps"
	"math/big"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/dense"
	"github.com/sHyanorn/retrieval-lab/internal/eval"
	"github.com/sHyanorn/retrieval-lab/internal/study"
)

// maxFile is the size below which a stored file must stay (amendment 9).
const maxFile = 50_000_000

// build is one graph: M, efConstruction and the seed of Section 6.
type build struct{ m, efc, seed int }

func (b build) String() string    { return fmt.Sprintf("m%d-efc%d-s%d", b.m, b.efc, b.seed) }
func (b build) run(ef int) string { return fmt.Sprintf("%s/ef%d", b, ef) }

// The builds of Section 6 searched over the whole grid: seeds 0 to 4 at
// M 16, required; M 8 and M 32 at seed 1, where they were built.
var (
	studyBuilds = []build{{16, 200, 0}, {16, 200, 1}, {16, 200, 2}, {16, 200, 3}, {16, 200, 4}}
	extraBuilds = []build{{8, 200, 1}, {32, 200, 1}}
	gridBuilds  = slices.Concat(studyBuilds, extraBuilds)
)

// The anchor build, searched at one ef.
var anchorBuild = build{16, 100, 0}

const anchorEF = 1000

// The fusions of Section 6, of the exact run and of seeds 1 to 4 at ef 100.
var fusions = []string{"rrf", "norm"}

// The recall targets of Sections 3 and 9.
var targets = []string{"0.80", "0.85", "0.90", "0.95", "0.99"}

func target(s string) *big.Rat {
	r, ok := new(big.Rat).SetString(s)
	if !ok {
		panic("bad target " + s)
	}
	return r
}

// confirmatorySeeds are the builds that H1 to H5 average over.
var confirmatorySeeds = []int{1, 2, 3, 4}

func confirmatory(seed int) build { return build{16, 200, seed} }

const (
	gridHeader   = "model\tdataset\tm\tefc\tseed\tef\tqueries\tndcg_cut_10\tnr_10\tnr_100"
	selectHeader = "model\tdataset\tm\tefc\tseed\ttarget\tef\treached\tstatus\tclosest_ef\tclosest_reached"
	storeHeader  = "run\tqid\tndcg_cut_10\trecall_100\tunjudged_10\tfound_10\tof_10\tfound_100\tof_100"
)

var (
	buildDir  = regexp.MustCompile(`^m(\d+)-efc(\d+)-s(\d+)$`)
	efFile    = regexp.MustCompile(`^ef(\d+)\.tsv$`)
	fusedFile = regexp.MustCompile(`^(rrf|norm)-(exact|s[1-4]-ef100)\.tsv$`)
)

// layout is what one model's directory on one dataset holds: the ef values
// of each build and the fused runs, checked against what the study needs.
type layout struct {
	dir    string
	builds map[build][]int
	fused  map[string]bool
	h4     []string // seeds 2 to 4 at seed 1's ef for recall 0.95
}

func readLayout(dir string) (*layout, error) {
	l := &layout{dir: dir, builds: map[build][]int{}, fused: map[string]bool{}}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	exact := false
	for _, e := range entries {
		name := e.Name()
		switch {
		case !e.IsDir() && name == "exact.tsv":
			exact = true
		case !e.IsDir() && fusedFile.MatchString(name):
			l.fused[strings.TrimSuffix(name, ".tsv")] = true
		case e.IsDir() && buildDir.MatchString(name):
			b, err := parseBuild(name)
			if err != nil {
				return nil, err
			}
			if !slices.Contains(gridBuilds, b) && b != anchorBuild {
				return nil, fmt.Errorf("%s: build %s is not one of the study", dir, name)
			}
			efs, err := readEFs(filepath.Join(dir, name))
			if err != nil {
				return nil, err
			}
			l.builds[b] = efs
		default:
			return nil, fmt.Errorf("%s: unexpected %s", dir, name)
		}
	}
	var missing []string
	if !exact {
		missing = append(missing, "exact.tsv")
	}
	for _, b := range gridBuilds {
		efs, ok := l.builds[b]
		if !ok && slices.Contains(extraBuilds, b) {
			continue
		}
		for _, ef := range dense.StudyEFs {
			if !slices.Contains(efs, ef) {
				missing = append(missing, b.run(ef)+".tsv")
			}
		}
		for _, ef := range efs {
			if !slices.Contains(dense.StudyEFs, ef) {
				return nil, fmt.Errorf("%s: %s is off the grid", dir, b.run(ef))
			}
		}
	}
	if efs, ok := l.builds[anchorBuild]; ok && !slices.Equal(efs, []int{anchorEF}) {
		return nil, fmt.Errorf("%s: the anchor build is searched at ef %d only, found %v", dir, anchorEF, efs)
	}
	for _, k := range fusions {
		for _, run := range fusedRuns(k) {
			if !l.fused[run] {
				missing = append(missing, run+".tsv")
			}
		}
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("%s: missing %s", dir, strings.Join(missing, ", "))
	}
	return l, nil
}

func fusedRuns(kind string) []string {
	out := []string{kind + "-exact"}
	for _, s := range confirmatorySeeds {
		out = append(out, fmt.Sprintf("%s-s%d-ef100", kind, s))
	}
	return out
}

func parseBuild(name string) (build, error) {
	m := buildDir.FindStringSubmatch(name)
	var v [3]int
	for i := range v {
		n, err := strconv.Atoi(m[i+1])
		if err != nil {
			return build{}, fmt.Errorf("build %s: %w", name, err)
		}
		v[i] = n
	}
	return build{v[0], v[1], v[2]}, nil
}

func readEFs(dir string) ([]int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var efs []int
	for _, e := range entries {
		m := efFile.FindStringSubmatch(e.Name())
		if e.IsDir() || m == nil {
			return nil, fmt.Errorf("%s: unexpected %s", dir, e.Name())
		}
		ef, err := strconv.Atoi(m[1])
		if err != nil {
			return nil, fmt.Errorf("%s: %w", filepath.Join(dir, e.Name()), err)
		}
		efs = append(efs, ef)
	}
	slices.Sort(efs)
	return efs, nil
}

// readQueries reads one per-query file and checks its qids against want,
// when given.
func readQueries(path string, want []string) ([]eval.Query, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close() //nolint:errcheck // read-only
	qs, err := eval.ReadPerQuery(bufio.NewReader(f))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if want != nil && !slices.Equal(qids(qs), want) {
		return nil, fmt.Errorf("%s: qids differ from those of exact.tsv", path)
	}
	return qs, nil
}

func qids(qs []eval.Query) []string {
	out := make([]string, len(qs))
	for i, q := range qs {
		out[i] = q.QID
	}
	return out
}

// gridCell is the means of one build at one ef (amendment 9).
type gridCell struct {
	ef      int
	queries int
	ndcg    float64
	nr10    *big.Rat
	nr100   *big.Rat // from ef 100 on
}

func measure(qs []eval.Query, ef int) (gridCell, error) {
	c := gridCell{ef: ef, queries: len(qs)}
	var err error
	if c.nr10, err = study.MeanRecall10(qs); err != nil {
		return c, err
	}
	if ef >= 100 {
		sum := new(big.Rat)
		for _, q := range qs {
			if q.Of100 <= 0 || q.Found100 < 0 || q.Found100 > q.Of100 {
				return c, fmt.Errorf("query %s: neighbour overlap at 100 %d of %d", q.QID, q.Found100, q.Of100)
			}
			sum.Add(sum, big.NewRat(int64(q.Found100), int64(q.Of100)))
		}
		c.nr100 = sum.Quo(sum, big.NewRat(int64(len(qs)), 1))
	}
	for _, q := range qs {
		c.ndcg += q.NDCG10
	}
	c.ndcg /= float64(len(qs))
	return c, nil
}

// collect reads the per-query measures under in, laid out as
// <model>/<dataset>/{exact.tsv, <build>/ef<ef>.tsv, <fusion>-exact.tsv,
// <fusion>-s<seed>-ef100.tsv}, and writes into out the means of every cell
// of the grid, the recall selections of every build, and per dataset and
// model the per-query measures of the points the readings use.
func collect(in, out string, limit int) error {
	models, err := subdirs(in)
	if err != nil {
		return err
	}
	if len(models) == 0 {
		return fmt.Errorf("%s: no models", in)
	}
	var grid, sel bytes.Buffer
	fmt.Fprintln(&grid, gridHeader)
	fmt.Fprintln(&sel, selectHeader)
	ref := map[string][]string{} // dataset → qids of the first model
	refModel := map[string]string{}
	for _, model := range models {
		datasets, err := subdirs(filepath.Join(in, model))
		if err != nil {
			return err
		}
		for _, ds := range datasets {
			l, err := readLayout(filepath.Join(in, model, ds))
			if err != nil {
				return err
			}
			exact, err := readQueries(filepath.Join(l.dir, "exact.tsv"), nil)
			if err != nil {
				return err
			}
			ids := qids(exact)
			if r, ok := ref[ds]; ok && !slices.Equal(r, ids) {
				return fmt.Errorf("%s: qids differ from those of %s", ds, refModel[ds])
			} else if !ok {
				ref[ds], refModel[ds] = ids, model
			}
			keep, err := l.cells(model, ds, ids, &grid, &sel)
			if err != nil {
				return err
			}
			if err := l.store(filepath.Join(out, ds, model+".tsv.gz"), exact, keep, limit); err != nil {
				return err
			}
		}
	}
	if err := os.WriteFile(filepath.Join(out, "grid.tsv"), grid.Bytes(), 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(out, "select.tsv"), sel.Bytes(), 0o644)
}

// cells writes the grid's means and the selections of one model on one
// dataset, and returns the runs to store, in order.
func (l *layout) cells(model, ds string, ids []string, grid, sel *bytes.Buffer) ([]string, error) {
	var keep []string
	for _, b := range append(slices.Clone(gridBuilds), anchorBuild) {
		efs, ok := l.builds[b]
		if !ok {
			continue
		}
		points := make([]study.Point, 0, len(efs))
		for _, ef := range efs {
			qs, err := readQueries(filepath.Join(l.dir, b.run(ef)+".tsv"), ids)
			if err != nil {
				return nil, err
			}
			c, err := measure(qs, ef)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", filepath.Join(l.dir, b.run(ef)+".tsv"), err)
			}
			nr100 := "-"
			if c.nr100 != nil {
				nr100 = c.nr100.RatString()
			}
			fmt.Fprintf(grid, "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\n", model, ds, b.m, b.efc, b.seed, ef,
				c.queries, strconv.FormatFloat(c.ndcg, 'g', -1, 64), c.nr10.RatString(), nr100)
			points = append(points, study.Point{EF: ef, Recall: c.nr10})
		}
		if b == anchorBuild {
			keep = append(keep, b.run(anchorEF))
			continue
		}
		set := map[int]bool{10: true, 40: true, 100: true}
		for _, t := range targets {
			s, err := study.Select(points, target(t))
			if err != nil {
				return nil, err
			}
			c, err := study.Closest(points, target(t))
			if err != nil {
				return nil, err
			}
			fmt.Fprintf(sel, "%s\t%s\t%d\t%d\t%d\t%s\t%d\t%s\t%s\t%d\t%s\n", model, ds, b.m, b.efc, b.seed, t,
				s.EF, s.Reached.RatString(), s.Status, c.EF, c.Recall.RatString())
			set[s.EF], set[c.EF] = true, true
			if b == confirmatory(1) && t == "0.95" {
				for _, seed := range confirmatorySeeds[1:] {
					// H4: seed 1's ef applied to the other builds; added
					// below, as those builds come after seed 1.
					l.h4 = append(l.h4, confirmatory(seed).run(s.EF))
				}
			}
		}
		for _, ef := range slices.Sorted(maps.Keys(set)) {
			keep = append(keep, b.run(ef))
		}
	}
	for _, r := range l.h4 {
		if !slices.Contains(keep, r) {
			keep = append(keep, r)
		}
	}
	for _, k := range fusions {
		keep = append(keep, fusedRuns(k)...)
	}
	return keep, nil
}

// store writes the runs of keep as float32 per-query measures, gzipped,
// refusing a file of limit bytes or more. Measures at depth 100 of a run
// cut at 10 hits (HNSW below ef 100, Section 6) are written as "-".
func (l *layout) store(path string, exact []eval.Query, keep []string, limit int) error {
	var text bytes.Buffer
	fmt.Fprintln(&text, storeHeader)
	ids := qids(exact)
	for _, run := range append([]string{"exact"}, keep...) {
		qs := exact
		if run != "exact" {
			var err error
			if qs, err = readQueries(filepath.Join(l.dir, run+".tsv"), ids); err != nil {
				return err
			}
		}
		shallow := false
		if _, ef, ok := strings.Cut(run, "/ef"); ok {
			n, err := strconv.Atoi(ef)
			if err != nil {
				return fmt.Errorf("run %s: %w", run, err)
			}
			shallow = n < 100
		}
		for _, q := range qs {
			r100, f100, o100 := f32(q.Recall100), strconv.Itoa(q.Found100), strconv.Itoa(q.Of100)
			if shallow {
				r100, f100, o100 = "-", "-", "-"
			}
			fmt.Fprintf(&text, "%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\t%s\n", run, q.QID, f32(q.NDCG10), r100, f32(q.Unjudged10),
				q.Found10, q.Of10, f100, o100)
		}
	}
	var buf bytes.Buffer
	z, err := gzip.NewWriterLevel(&buf, gzip.BestCompression)
	if err != nil {
		return err
	}
	if _, err := z.Write(text.Bytes()); err != nil {
		return err
	}
	if err := z.Close(); err != nil {
		return err
	}
	if buf.Len() >= limit {
		return fmt.Errorf("%s: %d bytes, over the limit of %d", path, buf.Len(), limit)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

func f32(x float64) string { return strconv.FormatFloat(float64(float32(x)), 'g', -1, 32) }

func subdirs(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if !e.IsDir() {
			return nil, fmt.Errorf("%s: unexpected %s", dir, e.Name())
		}
		out = append(out, e.Name())
	}
	return out, nil
}
