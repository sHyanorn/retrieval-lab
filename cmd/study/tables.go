package main

import (
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/study"
)

// point gives the ef at which a build is read at an operating point.
type point struct {
	name string
	ef   func(r *results, model, ds string, b build) (int, error)
}

func atEF(ef int) point {
	return point{fmt.Sprintf("ef %d", ef), func(*results, string, string, build) (int, error) { return ef, nil }}
}

func atRecall(t string, closest bool) point {
	name := "recall " + t
	if closest {
		name += ", closest"
	}
	return point{name, func(r *results, model, ds string, b build) (int, error) {
		s, err := r.selection(model, ds, b, t, closest)
		return s.EF, err
	}}
}

// change returns, per query, the mean over the confirmatory builds of f
// of the HNSW run at the point and the exact run.
func (r *results) change(model, ds string, p point, f func(a, e float64) float64) ([]float64, error) {
	exact, err := r.ndcg(model, ds, "exact")
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(exact))
	for _, s := range confirmatorySeeds {
		b := confirmatory(s)
		ef, err := p.ef(r, model, ds, b)
		if err != nil {
			return nil, err
		}
		x, err := r.ndcg(model, ds, b.run(ef))
		if err != nil {
			return nil, err
		}
		for q := range out {
			out[q] += f(x[q], exact[q])
		}
	}
	for q := range out {
		out[q] /= float64(len(confirmatorySeeds))
	}
	return out, nil
}

// strata lays out per-query values of the models on the datasets,
// checking that the models agree on each dataset's qids.
func (r *results) strata(models, datasets []string, value func(model, ds string) ([]float64, error)) (study.Strata, error) {
	s := make(study.Strata, len(datasets))
	for d, ds := range datasets {
		if err := r.aligned(models, ds); err != nil {
			return nil, err
		}
		for m, model := range models {
			v, err := value(model, ds)
			if err != nil {
				return nil, err
			}
			if m == 0 {
				s[d] = make([][]float64, len(v))
				for q := range v {
					s[d][q] = make([]float64, len(models))
				}
			}
			for q, x := range v {
				s[d][q][m] = x
			}
		}
	}
	return s, nil
}

// loss is L per query: −c̄, the change averaged over the confirmatory builds.
func (r *results) loss(models, datasets []string, p point) (study.Strata, error) {
	return r.strata(models, datasets, func(model, ds string) ([]float64, error) {
		return r.change(model, ds, p, func(a, e float64) float64 { return e - a })
	})
}

// approx is nDCG@10 under HNSW per query, averaged over the builds.
func (r *results) approx(models, datasets []string, p point) (study.Strata, error) {
	return r.strata(models, datasets, func(model, ds string) ([]float64, error) {
		return r.change(model, ds, p, func(a, _ float64) float64 { return a })
	})
}

func (r *results) exact(models, datasets []string) (study.Strata, error) {
	return r.strata(models, datasets, func(model, ds string) ([]float64, error) { return r.ndcg(model, ds, "exact") })
}

// fusedLoss is the loss of the fused run with HNSW at ef 100 against the
// fused run with exact search, averaged over the confirmatory builds.
func (r *results) fusedLoss(models, datasets []string, kind string) (study.Strata, error) {
	return r.strata(models, datasets, func(model, ds string) ([]float64, error) {
		e, err := r.ndcg(model, ds, kind+"-exact")
		if err != nil {
			return nil, err
		}
		out := make([]float64, len(e))
		for _, s := range confirmatorySeeds {
			a, err := r.ndcg(model, ds, fmt.Sprintf("%s-s%d-ef100", kind, s))
			if err != nil {
				return nil, err
			}
			for q := range out {
				out[q] += (e[q] - a[q]) / float64(len(confirmatorySeeds))
			}
		}
		return out, nil
	})
}

// sets are the models and datasets of the readings.
type sets struct {
	models    []string // every model stored, less those that failed a check
	cmodels   []string // the confirmatory models
	cdatasets []string // the confirmatory datasets
	failed    []string
}

// readingSets applies Section 3 with amendment 8: models that failed a
// check leave first, then the rule on NQ counts the models left on it.
func (r *results) readingSets(failed []string, tier1 bool) sets {
	all := map[string]bool{}
	encoded := map[string][]string{}
	for ds, cells := range r.cells {
		for m := range cells {
			all[m] = true
			encoded[ds] = append(encoded[ds], m)
		}
		slices.Sort(encoded[ds])
	}
	models := slices.Sorted(maps.Keys(all))
	cm, cd := study.Confirmatory(models, failed, encoded, tier1)
	return sets{
		models:    slices.DeleteFunc(slices.Clone(models), func(m string) bool { return slices.Contains(failed, m) }),
		cmodels:   cm,
		cdatasets: cd,
		failed:    failed,
	}
}

func (r *results) recallSet(models, datasets []string, t string) ([]string, error) {
	var err error
	s := study.RecallSet(models, datasets, func(m, ds string) bool {
		ok, e := r.matched(m, ds, t)
		if e != nil && err == nil {
			err = e
		}
		return ok
	})
	return s, err
}

// tables computes the readings from the results collect wrote and writes
// them between the markers of the document.
func tables(dir, doc string, failed []string, tier1 bool) error {
	r, err := readResults(dir)
	if err != nil {
		return err
	}
	conf, expl, err := r.report(r.readingSets(failed, tier1))
	if err != nil {
		return err
	}
	b, err := os.ReadFile(doc)
	if err != nil {
		return err
	}
	text := string(b)
	if text, err = splice(text, "confirmatory", conf); err != nil {
		return err
	}
	if text, err = splice(text, "exploratory", expl); err != nil {
		return err
	}
	return os.WriteFile(doc, []byte(text), 0o644)
}

// splice replaces the text between "<!-- name -->" and "<!-- /name -->".
func splice(text, name, table string) (string, error) {
	start, end := "<!-- "+name+" -->\n", "<!-- /"+name+" -->"
	i, j := strings.Index(text, start), strings.Index(text, end)
	if i < 0 || j < i {
		return "", fmt.Errorf("markers for %s not found", name)
	}
	return text[:i+len(start)] + table + text[j:], nil
}
