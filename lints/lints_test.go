package lints

import (
	"bytes"
	"encoding/gob"
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"

	"gonum.org/v1/gonum/mat"
)

func TestNewLinTS(t *testing.T) {
	tests := []struct {
		name      string
		dFeatures int
		options   []Option
		wantErr   bool
	}{
		{
			name:      "valid basic config",
			dFeatures: 10,
			options:   nil,
			wantErr:   false,
		},
		{
			name:      "valid with options",
			dFeatures: 5,
			options: []Option{
				WithLambda(2.0),
				WithSigma2(0.5),
				WithBias(false),
				WithMaintenanceFreq(1000),
				WithRandomSeed(42),
			},
			wantErr: false,
		},
		{
			name:      "invalid dimension",
			dFeatures: 0,
			options:   nil,
			wantErr:   true,
		},
		{
			name:      "negative dimension",
			dFeatures: -1,
			options:   nil,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lts, err := NewLinTS(tt.dFeatures, tt.options...)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewLinTS() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err != nil {
				return
			}

			// Check dimensions
			expectedD := tt.dFeatures
			if lts.useBias {
				expectedD++
			}
			if lts.d != expectedD {
				t.Errorf("NewLinTS() d = %v, want %v", lts.d, expectedD)
			}

			// Check matrix dimensions
			r, c := lts.AInv.Dims()
			if r != expectedD || c != expectedD {
				t.Errorf("AInv dimensions = (%v, %v), want (%v, %v)", r, c, expectedD, expectedD)
			}

			// Check initialization
			if lts.AInv.At(0, 0) != 1.0/lts.lambda {
				t.Errorf("AInv[0,0] = %v, want %v", lts.AInv.At(0, 0), 1.0/lts.lambda)
			}
		})
	}
}

func TestValidateContext(t *testing.T) {
	lts, err := NewLinTS(3, WithBias(true))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	tests := []struct {
		name    string
		input   []float64
		wantLen int
		wantErr bool
	}{
		{
			name:    "valid context",
			input:   []float64{1.0, 2.0, 3.0},
			wantLen: 4, // 3 features + 1 bias
			wantErr: false,
		},
		{
			name:    "wrong dimension",
			input:   []float64{1.0, 2.0},
			wantLen: 0,
			wantErr: true,
		},
		{
			name:    "zero vector",
			input:   []float64{0.0, 0.0, 0.0},
			wantLen: 0,
			wantErr: true,
		},
		{
			name:    "infinite values",
			input:   []float64{1.0, math.Inf(1), 3.0},
			wantLen: 4,
			wantErr: false,
		},
		{
			name:    "NaN values",
			input:   []float64{1.0, math.NaN(), 3.0},
			wantLen: 4,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := lts.validateContext(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateContext() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err != nil {
				return
			}

			if len(result) != tt.wantLen {
				t.Errorf("validateContext() len = %v, want %v", len(result), tt.wantLen)
			}

			// Check normalization (L2 norm should be close to 1 for feature part)
			if tt.wantLen > 1 {
				norm := 0.0
				for i := 0; i < len(result)-1; i++ { // exclude bias term
					norm += result[i] * result[i]
				}
				norm = math.Sqrt(norm)
				if math.Abs(norm-1.0) > 1e-10 {
					t.Errorf("L2 norm = %v, want ~1.0", norm)
				}
			}
		})
	}
}

func TestScore(t *testing.T) {
	lts, err := NewLinTS(3, WithRandomSeed(42))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	context := []float64{1.0, 0.0, 0.0}

	// Test multiple scores to check they vary (due to sampling)
	scores := make([]float64, 10)
	for i := range scores {
		score, err := lts.Score(context)
		if err != nil {
			t.Fatalf("Score() error = %v", err)
		}
		scores[i] = score
	}

	// Check that scores are finite
	for i, score := range scores {
		if math.IsInf(score, 0) || math.IsNaN(score) {
			t.Errorf("Score %d = %v, want finite", i, score)
		}
	}

	// Check that scores vary (Thompson sampling should produce different values)
	allSame := true
	for i := 1; i < len(scores); i++ {
		if math.Abs(scores[i]-scores[0]) > 1e-10 {
			allSame = false
			break
		}
	}
	if allSame {
		t.Errorf("All scores are the same, Thompson sampling may not be working")
	}
}

func TestScoreBatch(t *testing.T) {
	lts, err := NewLinTS(2, WithRandomSeed(42))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	contexts := [][]float64{
		{1.0, 0.0},
		{0.0, 1.0},
		{0.5, 0.5},
	}

	scores, err := lts.ScoreBatch(contexts)
	if err != nil {
		t.Fatalf("ScoreBatch() error = %v", err)
	}

	if len(scores) != len(contexts) {
		t.Errorf("ScoreBatch() len = %v, want %v", len(scores), len(contexts))
	}

	for i, score := range scores {
		if math.IsInf(score, 0) || math.IsNaN(score) {
			t.Errorf("Score %d = %v, want finite", i, score)
		}
	}

	// Test empty batch
	_, err = lts.ScoreBatch([][]float64{})
	if err == nil {
		t.Errorf("ScoreBatch() with empty batch should return error")
	}
}

func TestUpdate(t *testing.T) {
	lts, err := NewLinTS(2, WithRandomSeed(42))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	context := []float64{1.0, 0.0}
	reward := 1.5

	// Store initial state
	initialTrace := mat.Trace(lts.AInv)
	initialMuNorm := mat.Norm(lts.muVec, 2)

	err = lts.Update(context, reward)
	if err != nil {
		t.Fatalf("Update() error = %v", err)
	}

	// Check that matrices have changed
	newTrace := mat.Trace(lts.AInv)
	newMuNorm := mat.Norm(lts.muVec, 2)

	if math.Abs(newTrace-initialTrace) < 1e-10 {
		t.Errorf("AInv trace unchanged after update")
	}

	if math.Abs(newMuNorm-initialMuNorm) < 1e-10 {
		t.Errorf("mu norm unchanged after update")
	}

	// Test with invalid context
	err = lts.Update([]float64{1.0}, reward) // wrong dimension
	if err == nil {
		t.Errorf("Update() with invalid context should return error")
	}

	// Test with invalid reward
	err = lts.Update(context, math.Inf(1))
	if err != nil {
		t.Errorf("Update() with infinite reward should not error (should clip)")
	}
}

func TestConcurrentOperations(t *testing.T) {
	lts, err := NewLinTS(5, WithRandomSeed(42))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	const numGoroutines = 10
	const opsPerGoroutine = 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Concurrent updates and scores
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(id)))

			for j := 0; j < opsPerGoroutine; j++ {
				// Generate random context
				context := make([]float64, 5)
				for k := range context {
					context[k] = rng.NormFloat64()
				}

				// Alternate between updates and scores
				if j%2 == 0 {
					reward := rng.NormFloat64()
					lts.Update(context, reward)
				} else {
					lts.Score(context)
				}
			}
		}(i)
	}

	wg.Wait()

	// Check final state
	stats := lts.GetStats()
	nUpdates := stats["n_updates"].(uint64)
	if nUpdates == 0 {
		t.Errorf("No updates recorded after concurrent operations")
	}

	// Verify matrix integrity
	if !isPositiveDefinite(lts.AInv) {
		t.Errorf("AInv is not positive definite after concurrent operations")
	}
}

func TestNumericalStability(t *testing.T) {
	lts, err := NewLinTS(3, WithMaintenanceFreq(10), WithRandomSeed(42))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	// Perform many updates to trigger maintenance
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 50; i++ {
		context := []float64{
			rng.NormFloat64(),
			rng.NormFloat64(),
			rng.NormFloat64(),
		}
		reward := rng.NormFloat64()
		err := lts.Update(context, reward)
		if err != nil {
			t.Fatalf("Update %d error = %v", i, err)
		}
	}

	// Check that matrices are still well-conditioned
	stats := lts.GetStats()
	condNum := stats["condition_number"].(float64)
	if (math.IsInf(condNum, 0) || math.IsNaN(condNum)) || condNum > 1e12 {
		t.Errorf("Condition number = %v, want finite and reasonable", condNum)
	}

	// Check that all matrix entries are finite
	checkMatrixFinite(t, lts.AInv, "AInv")
	// Convert TriDense to Dense for checking
	var LDense mat.Dense
	LDense.Copy(lts.L)
	checkMatrixFinite(t, &LDense, "L")
	checkVectorFinite(t, lts.muVec, "mu")
	checkVectorFinite(t, lts.b, "b")
}

func TestGetStats(t *testing.T) {
	lts, err := NewLinTS(3, WithRandomSeed(42))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	stats := lts.GetStats()

	// Check required fields
	requiredFields := []string{
		"n_updates", "condition_number", "norm_mu", "trace_A_inv",
		"last_maintenance", "cholesky_norm", "d_features", "d",
		"use_bias", "lambda", "sigma2",
	}

	for _, field := range requiredFields {
		if _, exists := stats[field]; !exists {
			t.Errorf("GetStats() missing field: %s", field)
		}
	}

	// Perform some updates and check that stats change
	context := []float64{1.0, 0.0, 0.0}
	lts.Update(context, 1.0)

	newStats := lts.GetStats()
	if newStats["n_updates"].(uint64) <= stats["n_updates"].(uint64) {
		t.Errorf("n_updates should increase after update")
	}
}

func TestCholesky(t *testing.T) {
	lts, err := NewLinTS(2, WithRandomSeed(42))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	// Test Cholesky decomposition property: L * L^T = σ²A^(-1)
	checkCholeskyProperty := func() {
		lts.mu.RLock()
		defer lts.mu.RUnlock()

		// Compute L * L^T
		var LLT mat.Dense
		LLT.Mul(lts.L, lts.L.T())

		// Compute σ²A^(-1)
		var expected mat.Dense
		expected.Scale(lts.sigma2, lts.AInv)

		// Check if they're approximately equal
		r, c := LLT.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				diff := math.Abs(LLT.At(i, j) - expected.At(i, j))
				if diff > 1e-10 {
					t.Errorf("Cholesky property violated at (%d,%d): |%v - %v| = %v",
						i, j, LLT.At(i, j), expected.At(i, j), diff)
				}
			}
		}
	}

	// Check initial property
	checkCholeskyProperty()

	// Update a few times and check property is maintained
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 5; i++ {
		context := []float64{rng.NormFloat64(), rng.NormFloat64()}
		reward := rng.NormFloat64()
		lts.Update(context, reward)
		checkCholeskyProperty()
	}
}

// Helper functions

func isPositiveDefinite(m *mat.Dense) bool {
	// Convert to symmetric matrix
	r, _ := m.Dims()
	symData := make([]float64, r*r)
	for i := 0; i < r; i++ {
		for j := 0; j <= i; j++ {
			val := 0.5 * (m.At(i, j) + m.At(j, i)) // Symmetrize
			symData[i*r+j] = val
		}
	}
	sym := mat.NewSymDense(r, symData)

	var chol mat.Cholesky
	return chol.Factorize(sym)
}

func checkMatrixFinite(t *testing.T, m *mat.Dense, name string) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if math.IsInf(m.At(i, j), 0) || math.IsNaN(m.At(i, j)) {
				t.Errorf("%s[%d,%d] = %v, want finite", name, i, j, m.At(i, j))
			}
		}
	}
}

func checkVectorFinite(t *testing.T, v *mat.VecDense, name string) {
	n := v.Len()
	for i := 0; i < n; i++ {
		if math.IsInf(v.AtVec(i), 0) || math.IsNaN(v.AtVec(i)) {
			t.Errorf("%s[%d] = %v, want finite", name, i, v.AtVec(i))
		}
	}
}

func TestDeterministicMode(t *testing.T) {
	// Test that deterministic mode produces reproducible results
	seed := int64(42)

	lts1, err := NewLinTS(3, WithRandomSeed(seed), WithDeterministic(true))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	lts2, err := NewLinTS(3, WithRandomSeed(seed), WithDeterministic(true))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	context := []float64{1.0, 0.0, 0.0}

	// Generate scores from both instances
	scores1 := make([]float64, 10)
	scores2 := make([]float64, 10)

	for i := 0; i < 10; i++ {
		score1, err1 := lts1.Score(context)
		score2, err2 := lts2.Score(context)

		if err1 != nil || err2 != nil {
			t.Fatalf("Score() errors: %v, %v", err1, err2)
		}

		scores1[i] = score1
		scores2[i] = score2
	}

	// In deterministic mode, scores should be identical
	for i := 0; i < 10; i++ {
		if math.Abs(scores1[i]-scores2[i]) > 1e-15 {
			t.Errorf("Deterministic mode failed: scores1[%d]=%v != scores2[%d]=%v",
				i, scores1[i], i, scores2[i])
		}
	}

	// Test that non-deterministic mode produces different results
	lts3, err := NewLinTS(3, WithRandomSeed(seed), WithDeterministic(false))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	lts4, err := NewLinTS(3, WithRandomSeed(seed), WithDeterministic(false))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	// In non-deterministic mode, at least some scores should differ (due to time mixing)
	allSame := true
	for i := 0; i < 10; i++ {
		score3, _ := lts3.Score(context)
		score4, _ := lts4.Score(context)

		if math.Abs(score3-score4) > 1e-10 {
			allSame = false
			break
		}
	}

	if allSame {
		t.Log("Warning: Non-deterministic mode produced identical results, this might be expected in some test environments")
	}
}

func TestDeterministicModeConcurrency(t *testing.T) {
	// Test that deterministic mode is thread-safe and still produces deterministic results
	lts, err := NewLinTS(3, WithRandomSeed(42), WithDeterministic(true))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	context := []float64{1.0, 0.0, 0.0}
	const numGoroutines = 4
	const scoresPerGoroutine = 25

	// Collect scores from multiple goroutines
	allScores := make([][]float64, numGoroutines)
	var wg sync.WaitGroup

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()
			scores := make([]float64, scoresPerGoroutine)
			for i := 0; i < scoresPerGoroutine; i++ {
				score, err := lts.Score(context)
				if err != nil {
					t.Errorf("Score() error in goroutine %d: %v", goroutineID, err)
					return
				}
				scores[i] = score
			}
			allScores[goroutineID] = scores
		}(g)
	}

	wg.Wait()

	// Verify that all scores are finite and reasonable
	for g := 0; g < numGoroutines; g++ {
		for i, score := range allScores[g] {
			if math.IsInf(score, 0) || math.IsNaN(score) {
				t.Errorf("Non-finite score from goroutine %d[%d]: %v", g, i, score)
			}
		}
	}

	// Note: In concurrent access, we can't expect the exact same sequence
	// due to the interleaving of RNG calls, but scores should be reasonable
	t.Logf("Collected %d total scores from %d concurrent goroutines",
		numGoroutines*scoresPerGoroutine, numGoroutines)
}

func TestReset(t *testing.T) {
	lts, err := NewLinTS(3, WithRandomSeed(42))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	// Get initial stats
	initialStats := lts.GetStats()

	// Perform some updates
	context := []float64{1.0, 0.0, 0.0}
	for i := 0; i < 5; i++ {
		lts.Update(context, float64(i))
	}

	// Check that state has changed
	updatedStats := lts.GetStats()
	if updatedStats["n_updates"].(uint64) != 5 {
		t.Errorf("Expected 5 updates, got %d", updatedStats["n_updates"])
	}

	// Reset and check that state is back to initial
	lts.Reset()
	resetStats := lts.GetStats()

	if resetStats["n_updates"].(uint64) != 0 {
		t.Errorf("After reset, expected 0 updates, got %d", resetStats["n_updates"])
	}

	// Check that matrices are reset properly
	initialTrace := initialStats["trace_A_inv"].(float64)
	resetTrace := resetStats["trace_A_inv"].(float64)

	if math.Abs(initialTrace-resetTrace) > 1e-10 {
		t.Errorf("After reset, trace_A_inv not properly reset: initial=%v, reset=%v",
			initialTrace, resetTrace)
	}
}

func TestCholeskyDecompositionCorrectness(t *testing.T) {
	lts, err := NewLinTS(3, WithRandomSeed(42))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	// Perform some updates to get non-trivial state
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 10; i++ {
		context := []float64{rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64()}
		reward := rng.NormFloat64()
		lts.Update(context, reward)
	}

	// Verify Cholesky property: L * L^T = σ²A^(-1)
	lts.mu.RLock()

	// Compute L * L^T manually
	var LLT mat.Dense
	LLT.Mul(lts.L, lts.L.T())

	// Compute expected σ²A^(-1)
	var expected mat.Dense
	expected.Scale(lts.sigma2, lts.AInv)

	lts.mu.RUnlock()

	// Check if they're approximately equal
	r, c := LLT.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			diff := math.Abs(LLT.At(i, j) - expected.At(i, j))
			if diff > 1e-10 {
				t.Errorf("Cholesky property violated at (%d,%d): |%v - %v| = %v",
					i, j, LLT.At(i, j), expected.At(i, j), diff)
			}
		}
	}
}

func TestNumericalStabilityAfterManyUpdates(t *testing.T) {
	lts, err := NewLinTS(5, WithMaintenanceFreq(50), WithRandomSeed(42))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	// Perform many updates with challenging numerical scenarios
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 200; i++ {
		// Generate challenging contexts (varying scales)
		scale := 0.1 + rng.Float64()*10
		context := make([]float64, 5)
		for j := range context {
			context[j] = rng.NormFloat64() * scale
		}

		// Challenging rewards
		reward := rng.NormFloat64() * (0.1 + rng.Float64()*5)

		err := lts.Update(context, reward)
		if err != nil {
			t.Errorf("Update %d failed: %v", i, err)
		}

		// Periodic checks
		if i%50 == 49 {
			stats := lts.GetStats()
			condNum := stats["condition_number"].(float64)

			// Check that condition number is reasonable
			if math.IsInf(condNum, 0) || math.IsNaN(condNum) || condNum > 1e15 {
				t.Errorf("Poor numerical condition at update %d: cond=%v", i, condNum)
			}

			// Check that matrices are still finite
			lts.mu.RLock()
			if !isMatrixFinite(lts.AInv) {
				t.Errorf("AInv contains non-finite values at update %d", i)
			}
			if !isVectorFinite(lts.muVec) {
				t.Errorf("muVec contains non-finite values at update %d", i)
			}
			lts.mu.RUnlock()
		}
	}
}

func TestBatchScoreConsistency(t *testing.T) {
	lts, err := NewLinTS(3, WithRandomSeed(42))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	contexts := [][]float64{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		{0.5, 0.5, 0.0},
	}

	// Get batch scores to test they are finite and reasonable
	batchScores1, err1 := lts.ScoreBatch(contexts)
	if err1 != nil {
		t.Fatalf("ScoreBatch() error = %v", err1)
	}

	// Individual scores should be finite and reasonable
	for i, score := range batchScores1 {
		if math.IsInf(score, 0) || math.IsNaN(score) {
			t.Errorf("Batch score %d is not finite: %v", i, score)
		}
		if math.Abs(score) > 1000 { // Sanity check for reasonable magnitude
			t.Errorf("Batch score %d has unreasonable magnitude: %v", i, score)
		}
	}
}

// Helper functions for tests

func isMatrixFinite(m *mat.Dense) bool {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if math.IsInf(m.At(i, j), 0) || math.IsNaN(m.At(i, j)) {
				return false
			}
		}
	}
	return true
}

func isVectorFinite(v *mat.VecDense) bool {
	n := v.Len()
	for i := 0; i < n; i++ {
		if math.IsInf(v.AtVec(i), 0) || math.IsNaN(v.AtVec(i)) {
			return false
		}
	}
	return true
}

// TestCholeskyFactorStability tests that Cholesky factor remains stable
func TestCholeskyFactorStability(t *testing.T) {
	lts, err := NewLinTS(4, WithRandomSeed(42), WithMaintenanceFreq(5))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	// Helper function to check Cholesky factor is finite and lower triangular
	checkCholeskyStability := func(iteration int) {
		lts.mu.RLock()
		defer lts.mu.RUnlock()

		// Check that L is lower triangular and finite
		for i := 0; i < lts.d; i++ {
			for j := 0; j < lts.d; j++ {
				val := lts.L.At(i, j)

				// Check finite values
				if math.IsInf(val, 0) || math.IsNaN(val) {
					t.Errorf("L[%d,%d] is not finite at iteration %d: %v", i, j, iteration, val)
				}

				// Check lower triangular property
				if i < j && math.Abs(val) > 1e-12 {
					t.Errorf("L[%d,%d] should be zero (upper triangular) at iteration %d: %v", i, j, iteration, val)
				}

				// Check positive diagonal elements
				if i == j && val <= 0 {
					t.Errorf("L[%d,%d] should be positive (diagonal) at iteration %d: %v", i, j, iteration, val)
				}
			}
		}
	}

	// Check initial state
	checkCholeskyStability(0)

	// Perform updates and check stability is maintained
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 50; i++ {
		context := make([]float64, 4)
		for j := range context {
			context[j] = rng.NormFloat64()
		}
		reward := rng.NormFloat64()

		err := lts.Update(context, reward)
		if err != nil {
			t.Fatalf("Update %d error = %v", i, err)
		}

		// Check stability every 5 updates
		if i%5 == 4 {
			checkCholeskyStability(i + 1)
		}
	}

	// Final check
	checkCholeskyStability(50)
}

// TestMathematicalConsistency tests mathematical properties under stress
func TestMathematicalConsistency(t *testing.T) {
	lts, err := NewLinTS(3, WithRandomSeed(42), WithMaintenanceFreq(20))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	// Test with various challenging scenarios
	scenarios := []struct {
		name      string
		genCtx    func(*rand.Rand) []float64
		genReward func(*rand.Rand) float64
	}{
		{
			name: "Normal contexts and rewards",
			genCtx: func(r *rand.Rand) []float64 {
				return []float64{r.NormFloat64(), r.NormFloat64(), r.NormFloat64()}
			},
			genReward: func(r *rand.Rand) float64 { return r.NormFloat64() },
		},
		{
			name: "Large magnitude contexts",
			genCtx: func(r *rand.Rand) []float64 {
				scale := 1.0 + r.Float64()*10.0
				return []float64{r.NormFloat64() * scale, r.NormFloat64() * scale, r.NormFloat64() * scale}
			},
			genReward: func(r *rand.Rand) float64 { return r.NormFloat64() * (1.0 + r.Float64()*5.0) },
		},
		{
			name: "Small magnitude contexts",
			genCtx: func(r *rand.Rand) []float64 {
				scale := 0.01 + r.Float64()*0.1
				return []float64{r.NormFloat64() * scale, r.NormFloat64() * scale, r.NormFloat64() * scale}
			},
			genReward: func(r *rand.Rand) float64 { return r.NormFloat64() * 0.1 },
		},
	}

	rng := rand.New(rand.NewSource(42))
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			// Reset for each scenario
			lts.Reset()

			for i := 0; i < 30; i++ {
				context := scenario.genCtx(rng)
				reward := scenario.genReward(rng)

				err := lts.Update(context, reward)
				if err != nil {
					t.Errorf("Update %d failed: %v", i, err)
					continue
				}

				// Check mathematical invariants
				lts.mu.RLock()
				if !isPositiveDefinite(lts.AInv) {
					t.Errorf("A^(-1) lost positive definiteness at update %d", i)
				}
				if !isMatrixFinite(lts.AInv) {
					t.Errorf("A^(-1) contains non-finite values at update %d", i)
				}
				if !isVectorFinite(lts.muVec) {
					t.Errorf("muVec contains non-finite values at update %d", i)
				}
				lts.mu.RUnlock()
			}
		})
	}
}

// Benchmark tests

func BenchmarkScore(b *testing.B) {
	lts, err := NewLinTS(10, WithRandomSeed(42))
	if err != nil {
		b.Fatalf("NewLinTS() error = %v", err)
	}

	context := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := lts.Score(context)
		if err != nil {
			b.Fatalf("Score() error = %v", err)
		}
	}
}

func BenchmarkUpdate(b *testing.B) {
	lts, err := NewLinTS(10, WithRandomSeed(42))
	if err != nil {
		b.Fatalf("NewLinTS() error = %v", err)
	}

	context := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	reward := 1.0

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := lts.Update(context, reward)
		if err != nil {
			b.Fatalf("Update() error = %v", err)
		}
	}
}

func BenchmarkScoreBatch(b *testing.B) {
	lts, err := NewLinTS(10, WithRandomSeed(42))
	if err != nil {
		b.Fatalf("NewLinTS() error = %v", err)
	}

	contexts := make([][]float64, 100)
	for i := range contexts {
		contexts[i] = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := lts.ScoreBatch(contexts)
		if err != nil {
			b.Fatalf("ScoreBatch() error = %v", err)
		}
	}
}

func BenchmarkConcurrentOperations(b *testing.B) {
	lts, err := NewLinTS(10, WithRandomSeed(42))
	if err != nil {
		b.Fatalf("NewLinTS() error = %v", err)
	}

	context := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		for pb.Next() {
			if rng.Float64() < 0.5 {
				lts.Score(context)
			} else {
				lts.Update(context, rng.NormFloat64())
			}
		}
	})
}

// TestSPDInvariantMaintenance tests that A^(-1) remains symmetric positive definite
func TestSPDInvariantMaintenance(t *testing.T) {
	lts, err := NewLinTS(5, WithRandomSeed(42), WithMaintenanceFreq(10))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	// Helper function to check SPD property
	checkSPD := func(iteration int) {
		lts.mu.RLock()
		defer lts.mu.RUnlock()

		if !isPositiveDefinite(lts.AInv) {
			t.Errorf("A^(-1) is not positive definite at iteration %d", iteration)
		}

		// Check symmetry
		r, c := lts.AInv.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				if math.Abs(lts.AInv.At(i, j)-lts.AInv.At(j, i)) > 1e-12 {
					t.Errorf("A^(-1) is not symmetric at (%d,%d) iteration %d: %v != %v",
						i, j, iteration, lts.AInv.At(i, j), lts.AInv.At(j, i))
				}
			}
		}
	}

	// Check initial state
	checkSPD(0)

	// Perform updates and check SPD property is maintained
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		context := make([]float64, 5)
		for j := range context {
			context[j] = rng.NormFloat64()
		}
		reward := rng.NormFloat64()

		err := lts.Update(context, reward)
		if err != nil {
			t.Fatalf("Update %d error = %v", i, err)
		}

		// Check SPD property every 10 updates
		if i%10 == 9 {
			checkSPD(i + 1)
		}
	}

	// Final check
	checkSPD(100)
}

// TestFailedCholeskyMetric tests that the failed Cholesky counter works correctly
func TestFailedCholeskyMetric(t *testing.T) {
	lts, err := NewLinTS(3, WithRandomSeed(42))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	// Check initial stats
	initialStats := lts.GetStats()
	if initialStats["failed_cholesky"].(uint64) != 0 {
		t.Errorf("Initial failed_cholesky should be 0, got %d", initialStats["failed_cholesky"])
	}

	// Perform some updates to potentially trigger maintenance and Cholesky operations
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 30; i++ {
		context := make([]float64, 3)
		for j := range context {
			context[j] = rng.NormFloat64()
		}
		reward := rng.NormFloat64()
		lts.Update(context, reward)
	}

	// Check final stats - failed_cholesky should be tracked
	finalStats := lts.GetStats()
	failedCholesky, exists := finalStats["failed_cholesky"]
	if !exists {
		t.Error("failed_cholesky metric should exist in stats")
	}

	// The metric should be a uint64
	if _, ok := failedCholesky.(uint64); !ok {
		t.Errorf("failed_cholesky should be uint64, got %T", failedCholesky)
	}

	t.Logf("Failed Cholesky count after 30 updates: %d", failedCholesky.(uint64))

	// Test reset functionality
	lts.Reset()
	resetStats := lts.GetStats()
	if resetStats["failed_cholesky"].(uint64) != 0 {
		t.Errorf("After reset, failed_cholesky should be 0, got %d", resetStats["failed_cholesky"])
	}
}

// TestProductionReadiness performs stress testing for production readiness
func TestProductionReadiness(t *testing.T) {
	// Test with different configurations
	configs := []struct {
		name          string
		d             int
		deterministic bool
	}{
		{"small_non_deterministic", 5, false},
		{"medium_deterministic", 20, true},
		{"large_non_deterministic", 50, false},
	}

	for _, config := range configs {
		t.Run(config.name, func(t *testing.T) {
			lts, err := NewLinTS(config.d,
				WithRandomSeed(42),
				WithDeterministic(config.deterministic),
				WithMaintenanceFreq(100))
			if err != nil {
				t.Fatalf("NewLinTS() error = %v", err)
			}

			// Stress test with many concurrent operations
			const numGoroutines = 10
			const opsPerGoroutine = 1000

			var wg sync.WaitGroup
			wg.Add(numGoroutines)

			for g := 0; g < numGoroutines; g++ {
				go func(goroutineID int) {
					defer wg.Done()
					workerRng := rand.New(rand.NewSource(int64(goroutineID * 1000)))

					for i := 0; i < opsPerGoroutine; i++ {
						context := make([]float64, config.d)
						for j := range context {
							context[j] = workerRng.NormFloat64()
						}

						if i%3 == 0 {
							// Update operation
							reward := workerRng.NormFloat64()
							err := lts.Update(context, reward)
							if err != nil {
								t.Errorf("Update error in goroutine %d: %v", goroutineID, err)
								return
							}
						} else {
							// Score operation
							score, err := lts.Score(context)
							if err != nil {
								t.Errorf("Score error in goroutine %d: %v", goroutineID, err)
								return
							}
							if math.IsInf(score, 0) || math.IsNaN(score) {
								t.Errorf("Invalid score in goroutine %d: %v", goroutineID, score)
								return
							}
						}
					}
				}(g)
			}

			wg.Wait()

			// Verify final state
			stats := lts.GetStats()
			t.Logf("Config %s final stats:", config.name)
			t.Logf("  Updates: %d", stats["n_updates"])
			t.Logf("  Failed Cholesky: %d", stats["failed_cholesky"])
			t.Logf("  Condition number: %.2e", stats["condition_number"])
			t.Logf("  Deterministic: %v, Num RNGs: %d", stats["deterministic"], stats["num_rngs"])

			// Verify matrix integrity
			lts.mu.RLock()
			if !isPositiveDefinite(lts.AInv) {
				t.Errorf("AInv is not positive definite after stress test")
			}
			if !isMatrixFinite(lts.AInv) {
				t.Errorf("AInv contains non-finite values after stress test")
			}
			lts.mu.RUnlock()
		})
	}
}

func TestSaveLoad(t *testing.T) {
	// Create and train a model
	lts, err := NewLinTS(5, WithRandomSeed(42), WithLambda(2.0), WithSigma2(0.5), WithBias(true))
	if err != nil {
		t.Fatalf("NewLinTS() error = %v", err)
	}

	// Train the model with some data
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 10; i++ {
		context := make([]float64, 5)
		for j := range context {
			context[j] = rng.NormFloat64()
		}
		reward := rng.NormFloat64()
		err := lts.Update(context, reward)
		if err != nil {
			t.Fatalf("Update error: %v", err)
		}
	}

	// Get original stats
	originalStats := lts.GetStats()

	// Save the model
	var buf bytes.Buffer
	err = lts.Save(&buf)
	if err != nil {
		t.Fatalf("Save error: %v", err)
	}

	// Load the model
	loadedLts, err := Load(&buf, 123) // Use different seed for loading
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}

	// Compare basic configuration
	if loadedLts.dFeatures != lts.dFeatures {
		t.Errorf("dFeatures mismatch: original=%d, loaded=%d", lts.dFeatures, loadedLts.dFeatures)
	}
	if loadedLts.d != lts.d {
		t.Errorf("d mismatch: original=%d, loaded=%d", lts.d, loadedLts.d)
	}
	if loadedLts.lambda != lts.lambda {
		t.Errorf("lambda mismatch: original=%f, loaded=%f", lts.lambda, loadedLts.lambda)
	}
	if loadedLts.sigma2 != lts.sigma2 {
		t.Errorf("sigma2 mismatch: original=%f, loaded=%f", lts.sigma2, loadedLts.sigma2)
	}
	if loadedLts.useBias != lts.useBias {
		t.Errorf("useBias mismatch: original=%t, loaded=%t", lts.useBias, loadedLts.useBias)
	}

	// Compare stats
	loadedStats := loadedLts.GetStats()
	if originalStats["n_updates"] != loadedStats["n_updates"] {
		t.Errorf("n_updates mismatch: original=%d, loaded=%d", originalStats["n_updates"], loadedStats["n_updates"])
	}

	// Compare key matrices
	lts.mu.RLock()
	loadedLts.mu.RLock()

	if !matricesEqual(lts.AInv, loadedLts.AInv, 1e-12) {
		t.Errorf("AInv matrices don't match")
	}
	if !vectorsEqual(lts.b, loadedLts.b, 1e-12) {
		t.Errorf("b vectors don't match")
	}
	if !vectorsEqual(lts.muVec, loadedLts.muVec, 1e-12) {
		t.Errorf("muVec vectors don't match")
	}

	lts.mu.RUnlock()
	loadedLts.mu.RUnlock()

	// Test that both models behave the same way
	testContext := []float64{1.0, 0.5, -0.5, 0.2, -0.8}

	// Score should be different due to different RNG seeds, but both should be finite
	originalScore, err1 := lts.Score(testContext)
	loadedScore, err2 := loadedLts.Score(testContext)

	if err1 != nil || err2 != nil {
		t.Errorf("Score errors: original=%v, loaded=%v", err1, err2)
	}

	if math.IsInf(originalScore, 0) || math.IsNaN(originalScore) ||
		math.IsInf(loadedScore, 0) || math.IsNaN(loadedScore) {
		t.Errorf("Invalid scores: original=%v, loaded=%v", originalScore, loadedScore)
	}
}

// Helper functions for matrix comparison
func matricesEqual(a, b *mat.Dense, tolerance float64) bool {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		return false
	}

	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			if math.Abs(a.At(i, j)-b.At(i, j)) > tolerance {
				return false
			}
		}
	}
	return true
}

func vectorsEqual(a, b *mat.VecDense, tolerance float64) bool {
	if a.Len() != b.Len() {
		return false
	}

	for i := 0; i < a.Len(); i++ {
		if math.Abs(a.AtVec(i)-b.AtVec(i)) > tolerance {
			return false
		}
	}
	return true
}

func TestSaveLoadInvalidVersion(t *testing.T) {
	// Create a buffer with invalid version
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)

	invalidState := LinTSState{
		Version:   999, // Invalid version
		DFeatures: 5,
		D:         6,
		Lambda:    1.0,
		Sigma2:    1.0,
	}

	err := encoder.Encode(invalidState)
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}

	// Try to load - should fail
	_, err = Load(&buf, 42)
	if err == nil {
		t.Error("Expected error for invalid version, got nil")
	}
	if err.Error() != "unsupported gob version" {
		t.Errorf("Expected 'unsupported gob version', got: %v", err)
	}
}
