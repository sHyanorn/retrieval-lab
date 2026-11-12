package eval

import (
	"bytes"
	"reflect"
	"slices"
	"testing"
)

func TestUnjudgedCountsMissingAndNegativeGrades(t *testing.T) {
	judged := map[string]int{"a": 1, "b": 0, "c": -1}
	ranked := []Hit{{"a", 5}, {"b", 4}, {"c", 3}, {"d", 2}}
	// c (grade −1) and d (no judgment) are unjudged; b, judged 0, is not.
	// Four hits of a top 10 are divided by 10, as trec_eval's P@10.
	if got := Unjudged(judged, ranked, 10); got != 0.2 {
		t.Errorf("Unjudged@10 = %v, want 0.2", got)
	}
	if got := Unjudged(judged, ranked, 2); got != 0 {
		t.Errorf("Unjudged@2 = %v, want 0", got)
	}
}

func TestPerQueryMatchesEvaluateAndCountsOverlap(t *testing.T) {
	qrels := Qrels{"q1": {"a": 2, "b": 1}, "q2": {"x": 1}}
	run := Run{"q1": {{"b", 0.9}, {"z", 0.8}}, "q2": {{"y", 0.5}}}
	exact := Run{"q1": {{"a", 0.95}, {"b", 0.9}, {"c", 0.1}}, "q2": {{"x", 0.7}, {"y", 0.5}}}
	got := PerQuery(qrels, run, exact)
	ev := Evaluate(qrels, run)
	want := []Query{
		{QID: "q1", NDCG10: ev["q1"].NDCG10, Recall100: ev["q1"].Recall100, Unjudged10: 0.1, Found10: 1, Of10: 3, Found100: 1, Of100: 3},
		{QID: "q2", NDCG10: ev["q2"].NDCG10, Recall100: ev["q2"].Recall100, Unjudged10: 0.1, Found10: 1, Of10: 2, Found100: 1, Of100: 2},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("PerQuery = %+v\nwant %+v", got, want)
	}
	var b bytes.Buffer
	if err := WritePerQuery(&b, got); err != nil {
		t.Fatal(err)
	}
	back, err := ReadPerQuery(&b)
	if err != nil || !reflect.DeepEqual(back, got) {
		t.Fatalf("round trip = %+v, %v", back, err)
	}
}

func TestOverlapCountTakesTheTopKOfBoth(t *testing.T) {
	exact := []Hit{{"a", 0.9}, {"b", 0.8}, {"c", 0.7}}
	approx := []Hit{{"c", 0.7}, {"a", 0.9}, {"e", 0.95}}
	// approx's top 2 is {e, a}: one of exact's top 2, {a, b}.
	if f, o := OverlapCount(exact, approx, 2); f != 1 || o != 2 {
		t.Errorf("OverlapCount(k=2) = %d of %d, want 1 of 2", f, o)
	}
}

// tiedAtTen is an exact run whose 10th and 11th documents, z and a, have
// the same score: trec_eval ranks z 10th, its ID being the higher.
func tiedAtTen() []Hit {
	var exact []Hit
	for i, id := range []string{"d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9"} {
		exact = append(exact, Hit{id, 0.9 - 0.01*float64(i)})
	}
	return append(exact, Hit{"a", 0.5}, Hit{"z", 0.5}, Hit{"c", 0.4})
}

func TestOverlapCountCountsTiesAtK(t *testing.T) {
	exact := tiedAtTen()
	top9 := func(h Hit) []Hit { return append(slices.Clone(exact[:9]), h) }
	// A run that keeps a, not z, holds the same exact neighbours: of the two
	// tied at the cut, which one it keeps is not an error of the search.
	withA := top9(Hit{"a", 0.5})
	if f, o := OverlapCount(exact, withA, 10); f != 10 || o != 10 {
		t.Errorf("OverlapCount with the tied a = %d of %d, want 10 of 10", f, o)
	}
	if f, _ := overlapCount(exact, withA, 10, byRank); f == 10 {
		t.Error("the variant counting exact's top 10 by rank passed")
	}
	// A run that misses d9 and holds both a and z holds one of the tied
	// documents more than the exact top 10 has room for: no exact top 10
	// leaves out d9.
	both := append(slices.Clone(exact[:8]), Hit{"a", 0.5}, Hit{"z", 0.5})
	if f, _ := OverlapCount(exact, both, 10); f != 9 {
		t.Errorf("OverlapCount without d9, with a and z = %d, want 9", f)
	}
	if f, _ := overlapCount(exact, both, 10, uncapped); f == 9 {
		t.Error("the variant counting every tied document passed")
	}
	// A run that holds neither of the tied documents misses one, whatever
	// it holds below them.
	if f, _ := OverlapCount(exact, top9(Hit{"c", 0.4}), 10); f != 9 {
		t.Errorf("OverlapCount with c below the cut = %d, want 9", f)
	}
	// Nor does a score equal to the 10th count for a document the exact run
	// does not hold.
	if f, _ := OverlapCount(exact, top9(Hit{"x", 0.5}), 10); f != 9 {
		t.Errorf("OverlapCount with x, not in the exact run = %d, want 9", f)
	}
	// Overlap, M3's measure, keeps counting by rank.
	if got := Overlap(exact, withA, 10); got != 0.9 {
		t.Errorf("Overlap = %v, want 0.9", got)
	}
}
