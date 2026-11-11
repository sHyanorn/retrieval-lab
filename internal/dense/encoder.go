// Package dense embeds text with BGE-base-en-v1.5 on ONNX Runtime and ranks
// documents by exact inner product.
package dense

import (
	"cmp"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync/atomic"
	"unicode"

	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/hftokenizer"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/sHyanorn/retrieval-lab/internal/beir"
)

// Dim is the size of a BGE-base vector.
const Dim = 768

// QueryPrefix is the instruction Anserini's BgeBaseEn15Encoder puts before
// every query.
const QueryPrefix = "Represent this sentence for searching relevant passages: "

// Init loads the ONNX Runtime shared library at lib, once per process.
func Init(lib string) error {
	ort.SetSharedLibraryPath(lib)
	return ort.InitializeEnvironment()
}

// Encoder embeds text: WordPiece tokens with [CLS] and [SEP], at most
// maxLen of them, through the model; the output is the L2-normalised
// vector of the [CLS] position. It is safe for concurrent use; each call
// runs on one thread.
type Encoder struct {
	sess   *ort.DynamicAdvancedSession
	tok    *hftokenizer.Tokenizer
	sep    int
	maxLen int
}

// NewEncoder loads an ONNX model whose one input is input_ids and whose
// output is last_hidden_state, and a Hugging Face tokenizer.json.
func NewEncoder(modelPath, tokenizerPath string, maxLen int) (*Encoder, error) {
	tok, err := hftokenizer.NewFromFile(&api.Config{}, tokenizerPath)
	if err != nil {
		return nil, err
	}
	if err := tok.With(api.EncodeOptions{AddSpecialTokens: true}); err != nil {
		return nil, err
	}
	sep := tok.Encode("")
	if len(sep) != 2 {
		return nil, fmt.Errorf("tokenizer: empty text gives %v, want [CLS] [SEP]", sep)
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy() //nolint:errcheck // the session holds its own copy
	if err := opts.SetIntraOpNumThreads(1); err != nil {
		return nil, err
	}
	if err := opts.SetInterOpNumThreads(1); err != nil {
		return nil, err
	}
	sess, err := ort.NewDynamicAdvancedSession(modelPath, []string{"input_ids"}, []string{"last_hidden_state"}, opts)
	if err != nil {
		return nil, err
	}
	return &Encoder{sess: sess, tok: tok, sep: sep[1], maxLen: maxLen}, nil
}

// Close releases the model.
func (e *Encoder) Close() error { return e.sess.Destroy() }

// Tokens returns the model input for text: [CLS], the WordPiece IDs and
// [SEP], cut to maxLen with [SEP] kept last.
func (e *Encoder) Tokens(text string) []int64 {
	ids := e.tok.Encode(text)
	if len(ids) > e.maxLen {
		ids = append(ids[:e.maxLen-1], e.sep)
	}
	out := make([]int64, len(ids))
	for i, id := range ids {
		out[i] = int64(id)
	}
	return out
}

// Encode returns the normalised embedding of text.
func (e *Encoder) Encode(text string) ([]float32, error) {
	ids := e.Tokens(text)
	in, err := ort.NewTensor(ort.NewShape(1, int64(len(ids))), ids)
	if err != nil {
		return nil, err
	}
	defer in.Destroy() //nolint:errcheck // the run's error is the one to report
	out := []ort.Value{nil}
	if err := e.sess.Run([]ort.Value{in}, out); err != nil {
		return nil, err
	}
	defer out[0].Destroy() //nolint:errcheck // the data is copied out first
	h, ok := out[0].(*ort.Tensor[float32])
	if !ok || len(h.GetData()) != len(ids)*Dim {
		return nil, errors.New("model output is not a float32 [1, tokens, 768] tensor")
	}
	return normalize(h.GetData()[:Dim]), nil
}

// normalize is Anserini's DenseEncoder.normalize: v / (‖v‖ + 1e-12), with
// the squares summed in float32.
func normalize(v []float32) []float32 {
	var ss float32
	for _, x := range v {
		ss += float32(x * x)
	}
	d := float32(math.Sqrt(float64(ss))) + 1e-12
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x / d
	}
	return out
}

// DocText is the text encoded for a document: title and text joined by a
// space, as Pyserini's AutoDocumentEncoder joins them. Comparing vectors
// with Pyserini's prebuilt index settled this, and a length of 512 (D010).
// Characters of Unicode category C other than tab, newline and carriage
// return are dropped: Hugging Face's BertNormalizer drops Cc, Cf and Co
// before Pyserini's encoder sees the text, go-huggingface only Cc (D010).
func DocText(d beir.Doc) string {
	return strings.Map(func(r rune) rune {
		if r != '\t' && r != '\n' && r != '\r' && unicode.Is(unicode.C, r) {
			return -1
		}
		return r
	}, d.Title+" "+d.Text)
}

// EncodeAll embeds texts on workers goroutines and returns the vectors
// row by row in the order of texts. progress, if not nil, is called after
// each text with the number done. Fewer than one worker counts as one.
func (e *Encoder) EncodeAll(texts []string, workers int, progress func(done int)) ([]float32, error) {
	workers = max(workers, 1)
	out := make([]float32, len(texts)*Dim)
	next := make(chan int)
	errs := make(chan error, workers)
	var done atomic.Int64
	for range workers {
		go func() {
			for i := range next {
				v, err := e.Encode(texts[i])
				if err != nil {
					errs <- fmt.Errorf("text %d: %w", i, err)
					return
				}
				copy(out[i*Dim:], v)
				if n := done.Add(1); progress != nil {
					progress(int(n))
				}
			}
			errs <- nil
		}()
	}
	var err error
	for i := range texts {
		if err != nil {
			break
		}
		select {
		case next <- i:
		case err = <-errs:
			workers--
		}
	}
	close(next)
	for range workers {
		err = cmp.Or(err, <-errs)
	}
	return out, err
}
