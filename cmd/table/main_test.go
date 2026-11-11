package main

import "testing"

func TestAgreementFollowsSpec(t *testing.T) {
	for _, c := range []struct {
		ours, anchor float64
		want         string
	}{
		{0.67894, 0.6789, "all four decimals"},
		{0.6794, 0.6789, "three printed decimals"},
		{0.6739, 0.6789, "within 0.005"},
		{0.6738, 0.6789, "outside 0.005"},
	} {
		if got := agreement(delta(c.ours, c.anchor)); got != c.want {
			t.Errorf("%v against %v: %q, want %q", c.ours, c.anchor, got, c.want)
		}
	}
}

func TestSpliceReplacesOnlyTheMarkedBlock(t *testing.T) {
	got, err := splice("a\n<!-- x -->\nold\n<!-- /x -->\nb\n", "x", "new\n")
	if want := "a\n<!-- x -->\nnew\n<!-- /x -->\nb\n"; err != nil || got != want {
		t.Fatalf("splice = %q, %v; want %q", got, err, want)
	}
	if _, err := splice("no markers", "x", ""); err == nil {
		t.Fatal("want an error without markers")
	}
}
