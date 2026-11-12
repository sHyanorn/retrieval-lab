// Command study gathers the per-query measures of the registered study
// (docs/PREREGISTRATION.md) and computes its readings.
//
//	study collect -in DIR -out study/results
//	study tables -results study/results -doc docs/STUDY.md [-failed a,b] [-tier1]
//
// collect reads the output of cmd/eval -per-query for every run and writes
// the means of the whole grid, the recall selections of every build, and
// the per-query measures of the points the readings use (amendment 9).
// tables reads only what collect wrote and writes the readings of Section 9
// between the markers of the document.
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "study:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return errors.New("usage: study collect|tables [flags]")
	}
	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	switch args[0] {
	case "collect":
		in := fs.String("in", "", "directory of per-query measures, <model>/<dataset>/...")
		out := fs.String("out", "study/results", "directory to write into")
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		return collect(*in, *out, maxFile)
	case "tables":
		results := fs.String("results", "study/results", "directory collect wrote")
		doc := fs.String("doc", "docs/STUDY.md", "document with the markers")
		failed := fs.String("failed", "", "comma-separated models that failed a check of Section 5")
		tier1 := fs.Bool("tier1", false, "the study ran at tier 1 (Section 12)")
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		var f []string
		for m := range strings.SplitSeq(*failed, ",") {
			if m = strings.TrimSpace(m); m != "" {
				f = append(f, m)
			}
		}
		return tables(*results, *doc, f, *tier1)
	}
	return fmt.Errorf("unknown command %q", args[0])
}
