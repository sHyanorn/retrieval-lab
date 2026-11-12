package main

import (
	"fmt"
	"slices"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/study"
)

// deltas are δ of Section 8 and the exploratory readings at half and twice it.
var deltas = []float64{0.0025, study.Delta, 0.01}

// opResult is a comparison of the confirmatory models at one operating
// point, averaged over datasets, with H2 and H3 where they apply.
type opResult struct {
	point    point
	datasets []string
	cmp      *study.Comparison
	at40     *study.Comparison // over the same datasets, for H2's reading
	h2       *study.H2Result
	h3       *study.H3Result
}

// dsLabel is the label of one dataset in a per-dataset family
// (amendments 11 and 12).
type dsLabel struct {
	ds        string
	models    []string
	cmp       *study.Comparison
	unmatched []string
}

type reporter struct {
	r    *results
	s    sets
	at40 map[string]*study.Comparison
}

func (r *results) report(s sets) (conf, expl string, err error) {
	if len(s.cmodels) < 2 || len(s.cdatasets) == 0 {
		return "", "", fmt.Errorf("%d confirmatory models on %d datasets: nothing to compare", len(s.cmodels), len(s.cdatasets))
	}
	rp := &reporter{r: r, s: s, at40: map[string]*study.Comparison{}}
	return rp.run()
}

// comparison40 is the comparison at ef 40 over the datasets, computed once.
func (rp *reporter) comparison40(datasets []string) (*study.Comparison, error) {
	k := strings.Join(datasets, ",")
	if c, ok := rp.at40[k]; ok {
		return c, nil
	}
	loss, err := rp.r.loss(rp.s.cmodels, datasets, atEF(40))
	if err != nil {
		return nil, err
	}
	c := study.Compare(loss)
	rp.at40[k] = &c
	return &c, nil
}

// recallPoint reads H2 and H3 at a recall target: by the smallest ef over
// S_r (Section 3), or by the closest recall over every confirmatory
// dataset (amendment 10).
func (rp *reporter) recallPoint(t string, closest bool) (opResult, error) {
	r, s := rp.r, rp.s
	p := atRecall(t, closest)
	o := opResult{point: p, datasets: s.cdatasets}
	if !closest {
		var err error
		if o.datasets, err = r.recallSet(s.cmodels, s.cdatasets, t); err != nil {
			return o, err
		}
		if len(o.datasets) < 3 {
			o.h2 = &study.H2Result{Verdict: study.NotTestable}
			o.h3 = &study.H3Result{Verdict: study.NotTestable}
			return o, nil
		}
	}
	loss, err := r.loss(s.cmodels, o.datasets, p)
	if err != nil {
		return o, err
	}
	cmp := study.Compare(loss)
	if o.at40, err = rp.comparison40(o.datasets); err != nil {
		return o, err
	}
	h2 := study.H2(len(o.datasets), cmp, *o.at40, loss, study.Delta)
	exact, err := r.exact(s.cmodels, o.datasets)
	if err != nil {
		return o, err
	}
	approx, err := r.approx(s.cmodels, o.datasets, p)
	if err != nil {
		return o, err
	}
	h3, err := study.H3(len(o.datasets), exact, approx)
	if err != nil {
		return o, err
	}
	o.cmp, o.h2, o.h3 = &cmp, &h2, &h3
	return o, nil
}

func (rp *reporter) run() (conf, expl string, err error) {
	r, s := rp.r, rp.s
	h1, err := rp.comparison40(s.cdatasets)
	if err != nil {
		return "", "", err
	}
	at95, err := rp.recallPoint("0.95", false)
	if err != nil {
		return "", "", err
	}
	h4, err := rp.h4()
	if err != nil {
		return "", "", err
	}
	h5, err := rp.h5("rrf")
	if err != nil {
		return "", "", err
	}
	conf = rp.confirmatory(h1, at95, h4, h5)

	ops := []opResult{}
	for _, ef := range []int{10, 40} {
		o := opResult{point: atEF(ef), datasets: s.cdatasets, cmp: h1}
		if ef != 40 {
			loss, err := r.loss(s.cmodels, s.cdatasets, o.point)
			if err != nil {
				return "", "", err
			}
			c := study.Compare(loss)
			o.cmp = &c
		}
		ops = append(ops, o)
	}
	for _, closest := range []bool{false, true} {
		for _, t := range targets {
			o := at95
			if t != "0.95" || closest {
				if o, err = rp.recallPoint(t, closest); err != nil {
					return "", "", err
				}
			}
			ops = append(ops, o)
		}
	}
	var b strings.Builder
	b.WriteString(rp.opsTable(ops))
	for _, p := range []struct {
		pt point
		t  string
	}{{atEF(40), ""}, {atRecall("0.95", false), "0.95"}} {
		fams, err := rp.perDataset(p.pt, p.t)
		if err != nil {
			return "", "", err
		}
		b.WriteString(dsTable(p.pt, fams))
	}
	norm, err := rp.h5("norm")
	if err != nil {
		return "", "", err
	}
	fmt.Fprintf(&b, "\nNormalize fusion at ef 100, as H5 reads RRF: L_D %s, L_F %s, φ %s %s.\n",
		f4(norm.LD), f4(norm.LF), f2(norm.Phi), interval2(norm.PhiLo, norm.PhiHi))
	builds, err := rp.buildsTable()
	if err != nil {
		return "", "", err
	}
	b.WriteString(builds)
	measures, err := rp.measuresTable()
	if err != nil {
		return "", "", err
	}
	b.WriteString(measures)
	return conf, b.String(), nil
}

func (rp *reporter) h4() (study.H4Result, error) {
	r, s := rp.r, rp.s
	var cells []study.H4Cell
	for _, m := range s.cmodels {
		for _, ds := range s.cdatasets {
			sel, err := r.selection(m, ds, confirmatory(1), "0.95", false)
			if err != nil {
				return study.H4Result{}, err
			}
			c := study.H4Cell{Model: m, Dataset: ds, Seed1Matched: sel.Status == study.Matched}
			for _, seed := range confirmatorySeeds {
				for _, at := range []struct {
					ef  int
					out *[]float64
				}{{sel.EF, &c.Builds}, {40, &c.BuildsAt40}} {
					x, err := r.ndcg(m, ds, confirmatory(seed).run(at.ef))
					if err != nil {
						return study.H4Result{}, err
					}
					*at.out = append(*at.out, mean(x))
				}
			}
			cells = append(cells, c)
		}
	}
	return study.H4(cells, study.Delta), nil
}

func (rp *reporter) h5(kind string) (study.H5Result, error) {
	r, s := rp.r, rp.s
	dense, err := r.loss(s.cmodels, s.cdatasets, atEF(100))
	if err != nil {
		return study.H5Result{}, err
	}
	fused, err := r.fusedLoss(s.cmodels, s.cdatasets, kind)
	if err != nil {
		return study.H5Result{}, err
	}
	return study.H5(dense, fused), nil
}

// perDataset labels each dataset at the point: one Holm family over the
// pairs of every model stored on each confirmatory dataset, TREC-COVID a
// family of its own (amendment 11); at a recall target, only the models
// matched on the dataset (amendment 12).
func (rp *reporter) perDataset(p point, t string) ([][]dsLabel, error) {
	r := rp.r
	var fams [][]dsLabel
	for _, dss := range [][]string{study.ConfirmatoryDatasets, {"trec-covid"}} {
		var fam []dsLabel
		for _, ds := range dss {
			if _, ok := r.cells[ds]; !ok {
				continue
			}
			l := dsLabel{ds: ds}
			for _, m := range rp.s.models {
				if _, ok := r.cells[ds][m]; !ok {
					continue
				}
				if t != "" {
					ok, err := r.matched(m, ds, t)
					if err != nil {
						return nil, err
					}
					if !ok {
						l.unmatched = append(l.unmatched, m)
						continue
					}
				}
				l.models = append(l.models, m)
			}
			if len(l.models) >= 2 {
				loss, err := r.loss(l.models, []string{ds}, p)
				if err != nil {
					return nil, err
				}
				l.cmp = &study.Comparison{Means: loss.Means(), Pairs: study.PairTests(loss), RangeBound: study.RangeBound(loss)}
			}
			fam = append(fam, l)
		}
		var pairs [][]study.Pair
		for _, l := range fam {
			if l.cmp != nil {
				pairs = append(pairs, l.cmp.Pairs)
			}
		}
		if len(pairs) > 0 {
			study.HolmPairs(pairs...)
		}
		if len(fam) > 0 {
			fams = append(fams, fam)
		}
	}
	return fams, nil
}

// single is the loss of one build at one ef, averaged over the queries.
func (rp *reporter) single(m, ds string, b build, t string, ef int) (string, error) {
	r := rp.r
	if _, ok := r.grid[gridKey{m, ds, b}]; !ok {
		return "–", nil
	}
	if t != "" {
		sel, err := r.selection(m, ds, b, t, false)
		if err != nil {
			return "", err
		}
		ef = sel.EF
	}
	e, err := r.ndcg(m, ds, "exact")
	if err != nil {
		return "", err
	}
	a, err := r.ndcg(m, ds, b.run(ef))
	if err != nil {
		return "", err
	}
	return f4(mean(e) - mean(a)), nil
}

func (rp *reporter) datasets() []string {
	var out []string
	for _, ds := range append(slices.Clone(study.ConfirmatoryDatasets), "trec-covid") {
		if _, ok := rp.r.cells[ds]; ok {
			out = append(out, ds)
		}
	}
	for _, ds := range sortedKeys(rp.r.cells) {
		if !slices.Contains(out, ds) {
			out = append(out, ds)
		}
	}
	return out
}

// buildsTable gives each model's loss on each dataset for seed 0, the mean
// of seeds 1 to 4, and M 16, M 8 and M 32 at seed 1, at ef 40 and at recall
// 0.95, each build at its own ef (amendment 14).
func (rp *reporter) buildsTable() (string, error) {
	r := rp.r
	var rows [][]string
	for _, ds := range rp.datasets() {
		for _, m := range rp.s.models {
			if _, ok := r.cells[ds][m]; !ok {
				continue
			}
			row := []string{m, ds}
			for _, t := range []string{"", "0.95"} {
				p := atEF(40)
				if t != "" {
					p = atRecall(t, false)
				}
				for _, b := range []build{{16, 200, 0}, {}, {16, 200, 1}, {8, 200, 1}, {32, 200, 1}} {
					if b == (build{}) {
						loss, err := r.change(m, ds, p, func(a, e float64) float64 { return e - a })
						if err != nil {
							return "", err
						}
						row = append(row, f4(mean(loss)))
						continue
					}
					v, err := rp.single(m, ds, b, t, 40)
					if err != nil {
						return "", err
					}
					row = append(row, v)
				}
			}
			rows = append(rows, row)
		}
	}
	return "\nLoss L by build, averaged over queries: at ef 40, and at recall 0.95 with each build at its own ef. Seeds 1–4 is the mean of the confirmatory builds; M 16, M 8 and M 32 are seed 1 (amendment 14).\n\n" +
		table([]string{"Model", "Dataset", "ef 40: seed 0", "seeds 1–4", "M 16", "M 8", "M 32", "0.95: seed 0", "seeds 1–4", "M 16", "M 8", "M 32"}, rows), nil
}

// measuresTable gives the means of the other measures of Section 7.
func (rp *reporter) measuresTable() (string, error) {
	r := rp.r
	var rows [][]string
	for _, ds := range rp.datasets() {
		for _, m := range rp.s.models {
			if _, ok := r.cells[ds][m]; !ok {
				continue
			}
			e, err := r.ndcg(m, ds, "exact")
			if err != nil {
				return "", err
			}
			anchor := "–"
			if a, err := r.ndcg(m, ds, anchorBuild.run(anchorEF)); err == nil {
				anchor = f4(mean(a))
			}
			re, err := r.recall100(m, ds, "exact")
			if err != nil {
				return "", err
			}
			ue, err := r.unjudged(m, ds, "exact")
			if err != nil {
				return "", err
			}
			var ra, ua float64
			for _, s := range confirmatorySeeds {
				x, err := r.recall100(m, ds, confirmatory(s).run(100))
				if err != nil {
					return "", err
				}
				u, err := r.unjudged(m, ds, confirmatory(s).run(40))
				if err != nil {
					return "", err
				}
				ra += mean(x) / float64(len(confirmatorySeeds))
				ua += mean(u) / float64(len(confirmatorySeeds))
			}
			rows = append(rows, []string{m, ds, f4(mean(e)), anchor, f4(mean(re)), f4(ra), f4(mean(ue)), f4(ua)})
		}
	}
	return "\nMeans over queries. Anchor: M 16, efConstruction 100, seed 0, ef 1000. HNSW columns average seeds 1–4.\n\n" +
		table([]string{"Model", "Dataset", "nDCG@10 exact", "anchor", "R@100 exact", "R@100 ef 100", "unjudged@10 exact", "unjudged@10 ef 40"}, rows), nil
}

func mean(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}
