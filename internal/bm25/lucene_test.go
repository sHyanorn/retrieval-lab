package bm25

import (
	"math"
	"testing"
)

func TestNumFreeValuesFollowsFromEncoding(t *testing.T) {
	// SmallFloat: NUM_FREE_VALUES = 255 - longToInt4(Integer.MAX_VALUE).
	if got := 255 - longToInt4(math.MaxInt32); got != numFreeValues {
		t.Fatalf("255 - longToInt4(MaxInt32) = %d, want %d", got, numFreeValues)
	}
}

func TestLengthEncodingRoundTrip(t *testing.T) {
	for i := range 41 {
		if got := byte4ToInt(intToByte4(i)); got != i {
			t.Fatalf("length %d decodes to %d, want it exact", i, got)
		}
	}
	// 100 - 24 = 76 = 0b1001100 keeps its top four bits 0b1001 << 3 = 72: 24 + 72.
	for in, want := range map[int]int{41: 40, 100: 96} {
		if got := byte4ToInt(intToByte4(in)); got != want {
			t.Errorf("length %d decodes to %d, want %d", in, got, want)
		}
	}
	prev := 0
	for i := 0; i < 1<<20; i++ {
		got := byte4ToInt(intToByte4(i))
		if got > i || got < prev {
			t.Fatalf("length %d decodes to %d after %d: want monotone and never above the length", i, got, prev)
		}
		prev = got
	}
}

func TestIDF(t *testing.T) {
	// log(1 + (10 - 1 + 0.5) / (1 + 0.5)) = log(22/3).
	if got, want := idf(1, 10), float32(math.Log(22.0/3)); got != want {
		t.Fatalf("idf(1, 10) = %v, want %v", got, want)
	}
	// A term in every document still scores above zero.
	if got := idf(10, 10); got <= 0 {
		t.Fatalf("idf(10, 10) = %v, want > 0", got)
	}
}

func TestTermScoreSaturates(t *testing.T) {
	// avgdl 10 and a length of 10 make the length factor 1:
	// normInverse = 1 / (k1 · ((1 − b) + b)) = 1 / 0.9.
	inv := normCache(10)[intToByte4(10)]
	const w = 2
	for _, f := range []float32{1, 3, 50} {
		want := w * f / (f + 0.9)
		if got := termScore(w, f, inv); math.Abs(float64(got-want)) > 1e-6 {
			t.Errorf("termScore(freq %v) = %v, want %v", f, got, want)
		}
	}
	if got := termScore(w, 1e6, inv); got >= w || got < w*0.9999 {
		t.Errorf("termScore(freq 1e6) = %v, want just below the weight %v", got, w)
	}
	// A length of 40 against avgdl 10: freq / (freq + 0.9 · (0.6 + 0.4 · 4)).
	if got, want := termScore(w, 1, normCache(10)[intToByte4(40)]), float32(w/(1+0.9*2.2)); math.Abs(float64(got-want)) > 1e-6 {
		t.Errorf("termScore at length 40 = %v, want %v", got, want)
	}
}

func TestAdjustTies(t *testing.T) {
	// The example in Anserini's ScoreTiesAdjusterReranker, plus a third tie.
	s := []float32{23.439316, 22.087432, 22.087432, 22.087432, 21.602508}
	adjustTies(s)
	tie := float32(22.0874) // the subtractions below run in float32, as in Java
	want := []float32{23.4393, tie, tie - 1e-6*1, tie - 1e-6*2, 21.6025}
	for i := range s {
		if s[i] != want[i] {
			t.Fatalf("adjusted %v, want %v", s, want)
		}
	}
}
