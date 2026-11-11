package bm25

import (
	"math"
	"math/bits"
)

// Ports of Lucene 10.5.0 (org.apache.lucene.search.similarities.BM25Similarity,
// org.apache.lucene.util.SmallFloat) and of Anserini's
// ScoreTiesAdjusterReranker. Arithmetic is float32 where Lucene's is; the
// explicit float32 conversions keep the compiler from fusing a multiply and
// an add, which would skip a rounding Java performs.

// k1 and b are Anserini's BM25 defaults, which Pyserini's --bm25 keeps.
const k1, b float32 = 0.9, 0.4

// numFreeValues is SmallFloat.NUM_FREE_VALUES: lengths below it are stored
// exactly.
const numFreeValues = 24

// intToByte4 is SmallFloat.intToByte4, the one-byte, lossy encoding of a
// document length that Lucene stores as the norm.
func intToByte4(i int) uint8 {
	if i < numFreeValues {
		return uint8(i)
	}
	return uint8(numFreeValues + longToInt4(int64(i-numFreeValues)))
}

// longToInt4 keeps the four most significant bits of i and a shift.
func longToInt4(i int64) int {
	n := bits.Len64(uint64(i))
	if n < 4 {
		return int(i)
	}
	shift := n - 4
	return int(i>>shift)&0x07 | (shift+1)<<3
}

// byte4ToInt is SmallFloat.byte4ToInt, the length a norm byte decodes to.
func byte4ToInt(e uint8) int {
	i := int(e)
	if i < numFreeValues {
		return i
	}
	return int(min(numFreeValues+int4ToLong(i-numFreeValues), math.MaxInt32))
}

func int4ToLong(i int) int64 {
	m, shift := int64(i&0x07), i>>3-1
	if shift == -1 {
		return m
	}
	return (m | 0x08) << shift
}

// idf is BM25Similarity.idf.
func idf(docFreq, docCount int64) float32 {
	return float32(math.Log(1 + (float64(docCount-docFreq)+0.5)/(float64(docFreq)+0.5)))
}

// normCache is the table BM25Similarity.scorer builds: for each norm byte,
// 1 / (k1 · ((1 − b) + b · length / avgdl)).
func normCache(avgdl float32) [256]float32 {
	var c [256]float32
	for i := range c {
		c[i] = 1 / (k1 * ((1 - b) + float32(b*float32(byte4ToInt(uint8(i))))/avgdl))
	}
	return c
}

// termScore is BM25Scorer.score: weight is boost · idf. It equals
// weight · freq / (freq + k1 · (1 − b + b · length / avgdl)).
func termScore(weight, freq, normInverse float32) float32 {
	return weight - weight/(1+float32(freq*normInverse))
}

// adjustTies is Anserini's ScoreTiesAdjusterReranker, applied to scores in
// rank order: round each to four decimals, then lower a score that is not
// more than 1e-4 below its predecessor by 1e-6 per tie so far, so that
// trec_eval keeps the rank order instead of reordering ties by document ID.
func adjustTies(s []float32) {
	dup := 0
	for i := range s {
		s[i] = float32(math.Round(float64(s[i])*1e4) / 1e4)
		if i == 0 || s[i-1]-s[i] > 1e-4 {
			dup = 0
		} else {
			dup++
			s[i] -= float32(1e-6 * float32(dup))
		}
	}
}
