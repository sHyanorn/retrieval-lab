package dense

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// testEncoder needs the model and ONNX Runtime: RL_MODEL_DIR holds
// bge-base-en-v1.5-optimized.onnx and tokenizer.json (make model), RL_ORT_LIB
// is libonnxruntime.so. Without them the test is skipped, or fails when
// RL_REQUIRE_MODEL is set.
func testEncoder(t *testing.T) *Encoder {
	t.Helper()
	dir, lib := os.Getenv("RL_MODEL_DIR"), os.Getenv("RL_ORT_LIB")
	if dir == "" || lib == "" {
		if os.Getenv("RL_REQUIRE_MODEL") != "" {
			t.Fatal("RL_MODEL_DIR and RL_ORT_LIB must be set")
		}
		t.Skip("RL_MODEL_DIR or RL_ORT_LIB not set")
	}
	if err := Init(lib); err != nil {
		t.Fatal(err)
	}
	e, err := NewEncoder(filepath.Join(dir, "bge-base-en-v1.5-optimized.onnx"), filepath.Join(dir, "tokenizer.json"), 512)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { e.Close() }) //nolint:errcheck // test cleanup
	return e
}

func TestEncoderMatchesAnserini(t *testing.T) {
	e := testEncoder(t)
	raw, err := os.ReadFile("testdata/anserini-bge-base-en-v1.5.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Examples []struct {
			Text   string
			Vector []float32
		}
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	for _, ex := range ref.Examples {
		got, err := e.Encode(QueryPrefix + ex.Text)
		if err != nil {
			t.Fatal(err)
		}
		worst := 0.0
		for i := range got {
			worst = max(worst, math.Abs(float64(got[i]-ex.Vector[i])))
		}
		t.Logf("%.40q: largest difference %.3g", ex.Text, worst)
		// Anserini's own test allows 1e-4 per component.
		if worst > 1e-4 {
			t.Errorf("%.40q: a component differs by %.3g, want at most 1e-4", ex.Text, worst)
		}
	}
}
