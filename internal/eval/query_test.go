package eval

import (
	"bytes"
	"reflect"
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
