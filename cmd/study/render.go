package main

import (
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	"github.com/sHyanorn/retrieval-lab/internal/study"
)

func f4(x float64) string { return strconv.FormatFloat(x, 'f', 4, 64) }
func f2(x float64) string { return strconv.FormatFloat(x, 'f', 2, 64) }
func p2(x float64) string { return strconv.FormatFloat(x, 'g', 2, 64) }

func interval4(lo, hi float64) string { return "[" + f4(lo) + ", " + f4(hi) + "]" }
func interval2(lo, hi float64) string { return "[" + f2(lo) + ", " + f2(hi) + "]" }

func sortedKeys[V any](m map[string]V) []string { return slices.Sorted(maps.Keys(m)) }

func table(header []string, rows [][]string) string {
	var b strings.Builder
	line := func(v []string) { b.WriteString("| " + strings.Join(v, " | ") + " |\n") }
	line(header)
	sep := make([]string, len(header))
	for i := range sep {
		sep[i] = "---"
	}
	line(sep)
	for _, r := range rows {
		line(r)
	}
	return b.String()
}

var readings = map[study.Reading]string{
	study.ReadingRemoved:    "apart at ef 40 and some L̄ outside (−δ, δ): the dependence at equal ef does not survive matching on neighbour recall",
	study.ReadingSideBySide: "apart at ef 40 and every L̄ inside (−δ, δ): the losses differ at ef 40, and none reaches δ at recall 0.95",
	study.ReadingNeither:    "equivalent at ef 40: the loss does not depend on the model at either point",
	study.ReadingAt95Only:   "unresolved at ef 40: the losses agree within δ at recall 0.95; at ef 40 a dependence is neither shown nor excluded",
}

func rejected(ps []study.Pair) string {
	n := 0
	for _, p := range ps {
		if p.Reject {
			n++
		}
	}
	return fmt.Sprintf("%d of %d", n, len(ps))
}

func label(c *study.Comparison, delta float64) string {
	if c == nil {
		return "–"
	}
	return c.Label(delta).String()
}

func (rp *reporter) pairName(p study.Pair) string {
	return rp.s.cmodels[p.I] + " – " + rp.s.cmodels[p.J]
}

func (rp *reporter) confirmatory(h1 *study.Comparison, at95 opResult, h4 study.H4Result, h5 study.H5Result) string {
	s := rp.s
	var b strings.Builder
	fmt.Fprintf(&b, "Confirmatory models (%d): %s. Confirmatory datasets (%d): %s.", len(s.cmodels), strings.Join(s.cmodels, ", "),
		len(s.cdatasets), strings.Join(s.cdatasets, ", "))
	if len(s.failed) > 0 {
		fmt.Fprintf(&b, " Left out by a failed check of Section 5: %s.", strings.Join(s.failed, ", "))
	}
	b.WriteString("\n\n")
	s95 := strings.Join(at95.datasets, ", ")
	if s95 == "" {
		s95 = "none"
	}
	l40 := h1.Label(study.Delta)
	rows := [][]string{{"H1", study.H1(l40).String(), strings.Join(s.cdatasets, ", "),
		fmt.Sprintf("label %s; upper end of R̄'s 90%% interval %s; pairs rejected %s", l40, f4(h1.RangeBound), rejected(h1.Pairs))}}
	h2 := at95.h2
	beside := ""
	if at95.cmp != nil {
		beside = fmt.Sprintf("label %s; upper end of R̄'s 90%% interval %s; H1's label over S_0.95 %s",
			at95.cmp.Label(study.Delta), f4(at95.cmp.RangeBound), at95.at40.Label(study.Delta))
		if h2.Verdict == study.Supported {
			beside += "; " + readings[h2.Reading]
			if h2.Negligible {
				beside += "; every L̄ at 0.95 inside (−δ, δ): every model's loss is negligible there"
			}
		}
	}
	rows = append(rows, []string{"H2", h2.Verdict.String(), s95, beside})
	h3 := at95.h3
	beside = ""
	if at95.cmp != nil {
		var taus []string
		for d, t := range h3.TauByDataset {
			taus = append(taus, at95.datasets[d]+" "+f2(t))
		}
		beside = fmt.Sprintf("exact search orders %d of %d pairs; reversed %d; ordered by exact search and not by HNSW %d; τ-b %s (per dataset: %s)",
			h3.Ordered, len(h3.Exact), h3.Reversed, h3.Unordered, f2(h3.Tau), strings.Join(taus, ", "))
	}
	rows = append(rows, []string{"H3", h3.Verdict.String(), s95, beside})
	var h4ds []string
	maxR, maxR40 := 0.0, 0.0
	for i, c := range h4.Cells {
		if !slices.Contains(h4ds, c.Dataset) {
			h4ds = append(h4ds, c.Dataset)
		}
		maxR, maxR40 = max(maxR, h4.Ranges[i]), max(maxR40, h4.Ranges40[i])
	}
	rows = append(rows, []string{"H4", h4.Verdict.String(), strings.Join(h4ds, ", "),
		fmt.Sprintf("%d cells with seed 1 matched; largest range %s; at ef 40 %s; four builds understate the spread of all builds",
			len(h4.Cells), f4(maxR), f4(maxR40))})
	rows = append(rows, []string{"H5", h5.Verdict.String(), strings.Join(s.cdatasets, ", "),
		fmt.Sprintf("L_D %s %s; L_F %s; φ %s %s (95%%)", f4(h5.LD), interval4(h5.LDLo, h5.LDHi), f4(h5.LF), f2(h5.Phi), interval2(h5.PhiLo, h5.PhiHi))})
	b.WriteString(table([]string{"Hypothesis", "Verdict", "Datasets", "Reported beside"}, rows))

	b.WriteString("\nAveraged losses L̄ of the confirmatory models: at ef 40 over the confirmatory datasets, at recall 0.95 over S_0.95.\n\n")
	rows = nil
	for m, model := range s.cmodels {
		r := []string{model, f4(h1.Means[m]), "–", "–"}
		if at95.cmp != nil {
			r[2] = f4(at95.cmp.Means[m])
		}
		if h2.Verdict == study.Supported {
			r[3] = interval4(h2.Intervals[m][0], h2.Intervals[m][1])
		}
		rows = append(rows, r)
	}
	b.WriteString(table([]string{"Model", "L̄ at ef 40", "L̄ at 0.95", "90% interval at 0.95"}, rows))

	b.WriteString("\nPairs: Ē and the randomization test's p, Holm over the pairs at each point.\n\n")
	rows = nil
	for i, p := range h1.Pairs {
		r := []string{rp.pairName(p), f4(p.Diff), p2(p.P), yes(p.Reject), "–", "–", "–"}
		if at95.cmp != nil {
			q := at95.cmp.Pairs[i]
			r[4], r[5], r[6] = f4(q.Diff), p2(q.P), yes(q.Reject)
		}
		rows = append(rows, r)
	}
	b.WriteString(table([]string{"Pair", "Ē ef 40", "p", "rejected", "Ē 0.95", "p", "rejected"}, rows))

	if at95.cmp != nil {
		b.WriteString("\nH3: difference in mean nDCG@10 of each pair, exact search and HNSW at each build's ef for 0.95, each search its own Holm family.\n\n")
		rows = nil
		for i, p := range h3.Exact {
			q := h3.Approx[i]
			rows = append(rows, []string{rp.pairName(p), f4(p.Diff), p2(p.P), yes(p.Reject), f4(q.Diff), p2(q.P), yes(q.Reject)})
		}
		b.WriteString(table([]string{"Pair", "exact", "p", "ordered", "HNSW", "p", "ordered"}, rows))
	}

	b.WriteString("\nH4: range of mean nDCG@10 over seeds 1–4 at seed 1's ef for 0.95, and at ef 40, in each cell where seed 1 is matched.\n\n")
	rows = nil
	for i, c := range h4.Cells {
		rows = append(rows, []string{c.Model, c.Dataset, f4(h4.Ranges[i]), f4(h4.Ranges40[i])})
	}
	b.WriteString(table([]string{"Model", "Dataset", "range", "range at ef 40"}, rows))
	return b.String()
}

func yes(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

func (rp *reporter) opsTable(ops []opResult) string {
	var rows [][]string
	for _, o := range ops {
		ds := strings.Join(o.datasets, ", ")
		if ds == "" {
			ds = "none"
		} else if slices.Equal(o.datasets, rp.s.cdatasets) {
			ds = fmt.Sprintf("all %d", len(o.datasets))
		}
		r := []string{o.point.name, ds}
		for _, d := range deltas {
			r = append(r, label(o.cmp, d))
		}
		bound, rej := "–", "–"
		if o.cmp != nil {
			bound, rej = f4(o.cmp.RangeBound), rejected(o.cmp.Pairs)
		}
		r = append(r, bound, rej)
		h2, h3, rev, tau := "–", "–", "–", "–"
		if o.h2 != nil {
			h2, h3 = o.h2.Verdict.String(), o.h3.Verdict.String()
		}
		if o.cmp != nil && o.h3 != nil {
			rev, tau = fmt.Sprintf("%d, %d", o.h3.Reversed, o.h3.Unordered), f2(o.h3.Tau)
		}
		rows = append(rows, append(r, h2, h3, rev, tau))
	}
	return "Operating points, the confirmatory models averaged over datasets. At a recall target, smallest ef over S_r (Section 3) or closest recall over every confirmatory dataset (amendment 10). H2 and H3 are read by their rules at each target (amendment 14).\n\n" +
		table([]string{"Point", "Datasets", "label δ 0.0025", "δ 0.005", "δ 0.01", "R̄ bound", "pairs rejected", "H2", "H3", "reversed, unordered", "τ-b"}, rows)
}

func dsTable(p point, fams [][]dsLabel) string {
	var b strings.Builder
	var sizes []string
	for _, fam := range fams {
		n := 0
		for _, l := range fam {
			if l.cmp != nil {
				n += len(l.cmp.Pairs)
			}
		}
		sizes = append(sizes, fmt.Sprintf("%d pairs over %d datasets", n, len(fam)))
	}
	fmt.Fprintf(&b, "\nLabels per dataset at %s, Holm families: %s.\n\n", p.name, strings.Join(sizes, "; "))
	var rows [][]string
	for _, fam := range fams {
		for _, l := range fam {
			r := []string{l.ds, strconv.Itoa(len(l.models))}
			for _, d := range deltas {
				r = append(r, label(l.cmp, d))
			}
			bound, rej := "–", "–"
			if l.cmp != nil {
				bound, rej = f4(l.cmp.RangeBound), rejected(l.cmp.Pairs)
			}
			un := strings.Join(l.unmatched, ", ")
			if un == "" {
				un = "–"
			}
			rows = append(rows, append(r, bound, rej, un))
		}
	}
	b.WriteString(table([]string{"Dataset", "models", "label δ 0.0025", "δ 0.005", "δ 0.01", "R̄ bound", "pairs rejected", "not matched"}, rows))
	return b.String()
}
