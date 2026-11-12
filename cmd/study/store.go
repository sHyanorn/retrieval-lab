package main

import (
	"bufio"
	"compress/gzip"
	"errors"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/study"
)

// row is one query's measures in a stored run; deep is false where the
// run was cut at 10 hits and its measures at depth 100 are not kept.
type row struct {
	ndcg, recall100, unjudged float64
	deep                      bool
}

// cell is one model on one dataset as stored: the qids, in order, and the
// measures of each run, aligned with them.
type cell struct {
	qids []string
	runs map[string][]row
}

type gridKey struct {
	model, dataset string
	b              build
}

// results is what collect wrote: the cells by dataset and model, and the
// mean neighbour recall@10 of every build at every ef.
type results struct {
	cells map[string]map[string]*cell
	grid  map[gridKey][]study.Point
}

func readResults(dir string) (*results, error) {
	r := &results{cells: map[string]map[string]*cell{}, grid: map[gridKey][]study.Point{}}
	if err := r.readGrid(filepath.Join(dir, "grid.tsv")); err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		files, err := filepath.Glob(filepath.Join(dir, e.Name(), "*.tsv.gz"))
		if err != nil {
			return nil, err
		}
		r.cells[e.Name()] = map[string]*cell{}
		for _, f := range files {
			c, err := readCell(f)
			if err != nil {
				return nil, err
			}
			r.cells[e.Name()][strings.TrimSuffix(filepath.Base(f), ".tsv.gz")] = c
		}
	}
	return r, nil
}

func (r *results) readGrid(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close() //nolint:errcheck // read-only
	sc := bufio.NewScanner(f)
	if !sc.Scan() || sc.Text() != gridHeader {
		return fmt.Errorf("%s: header is not %q", path, gridHeader)
	}
	for n := 2; sc.Scan(); n++ {
		v := strings.Split(sc.Text(), "\t")
		if len(v) != 10 {
			return fmt.Errorf("%s line %d: %d fields", path, n, len(v))
		}
		var nums [4]int
		for i := range nums {
			if nums[i], err = strconv.Atoi(v[2+i]); err != nil {
				return fmt.Errorf("%s line %d: %w", path, n, err)
			}
		}
		rec, ok := new(big.Rat).SetString(v[8])
		if !ok {
			return fmt.Errorf("%s line %d: recall %q", path, n, v[8])
		}
		k := gridKey{v[0], v[1], build{nums[0], nums[1], nums[2]}}
		r.grid[k] = append(r.grid[k], study.Point{EF: nums[3], Recall: rec})
	}
	return sc.Err()
}

func readCell(path string) (*cell, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close() //nolint:errcheck // read-only
	z, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	sc := bufio.NewScanner(z)
	if !sc.Scan() || sc.Text() != storeHeader {
		return nil, fmt.Errorf("%s: header is not %q", path, storeHeader)
	}
	c := &cell{runs: map[string][]row{}}
	ids := map[string][]string{}
	for n := 2; sc.Scan(); n++ {
		v := strings.Split(sc.Text(), "\t")
		if len(v) != 9 {
			return nil, fmt.Errorf("%s line %d: %d fields", path, n, len(v))
		}
		var q row
		var errs [3]error
		q.ndcg, errs[0] = strconv.ParseFloat(v[2], 32)
		q.unjudged, errs[1] = strconv.ParseFloat(v[4], 32)
		if q.deep = v[3] != "-"; q.deep {
			q.recall100, errs[2] = strconv.ParseFloat(v[3], 32)
		}
		if err := errors.Join(errs[:]...); err != nil {
			return nil, fmt.Errorf("%s line %d: %w", path, n, err)
		}
		c.runs[v[0]] = append(c.runs[v[0]], q)
		ids[v[0]] = append(ids[v[0]], v[1])
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	c.qids = ids["exact"]
	if c.qids == nil {
		return nil, fmt.Errorf("%s: no exact run", path)
	}
	for run, q := range ids {
		if !slices.Equal(q, c.qids) {
			return nil, fmt.Errorf("%s: qids of %s differ from those of the exact run", path, run)
		}
	}
	return c, nil
}

// get returns one stored run of a model on a dataset.
func (r *results) get(model, ds, run string) ([]row, error) {
	c, ok := r.cells[ds][model]
	if !ok {
		return nil, fmt.Errorf("%s on %s: not stored", model, ds)
	}
	rows, ok := c.runs[run]
	if !ok {
		return nil, fmt.Errorf("%s on %s: run %s not stored", model, ds, run)
	}
	return rows, nil
}

func (r *results) ndcg(model, ds, run string) ([]float64, error) {
	rows, err := r.get(model, ds, run)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(rows))
	for i, q := range rows {
		out[i] = q.ndcg
	}
	return out, nil
}

// recall100 returns Recall@100 of a run that holds 100 hits (Section 7).
func (r *results) recall100(model, ds, run string) ([]float64, error) {
	rows, err := r.get(model, ds, run)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(rows))
	for i, q := range rows {
		if !q.deep {
			return nil, fmt.Errorf("%s on %s: run %s holds 10 hits, not 100", model, ds, run)
		}
		out[i] = q.recall100
	}
	return out, nil
}

func (r *results) unjudged(model, ds, run string) ([]float64, error) {
	rows, err := r.get(model, ds, run)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(rows))
	for i, q := range rows {
		out[i] = q.unjudged
	}
	return out, nil
}

// aligned checks that the models are stored on the dataset with the same
// qids, in the same order.
func (r *results) aligned(models []string, ds string) error {
	var first *cell
	for _, m := range models {
		c, ok := r.cells[ds][m]
		if !ok {
			return fmt.Errorf("%s on %s: not stored", m, ds)
		}
		if first == nil {
			first = c
		} else if !slices.Equal(c.qids, first.qids) {
			return fmt.Errorf("%s: qids of %s differ from those of %s", ds, m, models[0])
		}
	}
	return nil
}

// selection is the ef of a build at a recall target, by the rule of
// Section 3 (smallest) or of amendment 10 (closest).
func (r *results) selection(model, ds string, b build, t string, closest bool) (study.Selection, error) {
	points, ok := r.grid[gridKey{model, ds, b}]
	if !ok {
		return study.Selection{}, fmt.Errorf("%s on %s: build %s not in the grid", model, ds, b)
	}
	if closest {
		p, err := study.Closest(points, target(t))
		return study.Selection{EF: p.EF, Reached: p.Recall}, err
	}
	return study.Select(points, target(t))
}

// matched reports whether a model is matched on a dataset at a target:
// each of the confirmatory builds is.
func (r *results) matched(model, ds, t string) (bool, error) {
	var sels []study.Selection
	for _, s := range confirmatorySeeds {
		sel, err := r.selection(model, ds, confirmatory(s), t, false)
		if err != nil {
			return false, err
		}
		sels = append(sels, sel)
	}
	return study.AllMatched(sels), nil
}
