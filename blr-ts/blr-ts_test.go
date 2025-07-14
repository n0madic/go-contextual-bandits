package blrts

import (
	"bytes"
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestBLRTSInitialization(t *testing.T) {
	const (
		nArms      = 3
		contextDim = 2
		alpha      = 1.0
		sigma      = 0.1
		seed       = 42
		tol        = 1e-10
	)

	b, err := NewBLRTS(nArms, contextDim, alpha, sigma, seed)
	if err != nil {
		t.Fatalf("Failed to create BLRTS: %v", err)
	}

	if b.nArms != nArms || b.contextDim != contextDim || b.alpha != alpha || b.sigma != sigma {
		t.Errorf("Initialization parameters mismatch")
	}

	for a := range nArms {
		// Check mu is zero vector
		for i := 0; i < contextDim; i++ {
			if math.Abs(b.mu[a].AtVec(i)-0) > tol {
				t.Errorf("Mu not initialized to zero for arm %d at %d: got %f", a, i, b.mu[a].AtVec(i))
			}
		}

		// Check cov is alpha * I
		for i := range contextDim {
			for j := range contextDim {
				expected := 0.0
				if i == j {
					expected = alpha
				}
				if math.Abs(b.cov[a].At(i, j)-expected) > tol {
					t.Errorf("Cov not initialized correctly for arm %d at (%d,%d): got %f, want %f", a, i, j, b.cov[a].At(i, j), expected)
				}
			}
		}

		// Check chol is Cholesky of alpha * I (sqrt(alpha) on diagonal, zeros elsewhere)
		sqrtAlpha := math.Sqrt(alpha)
		for i := range contextDim {
			for j := range contextDim {
				expected := 0.0
				if i == j {
					expected = sqrtAlpha
				}
				got := b.chol[a].At(i, j)
				if math.Abs(got-expected) > tol {
					t.Errorf("Chol not initialized correctly for arm %d at (%d,%d): got %f, want %f", a, i, j, got, expected)
				}
			}
		}
	}
}

func TestBLRTSSelectActionAndUpdate(t *testing.T) {
	const (
		nArms      = 3
		contextDim = 2
		alpha      = 1.0
		sigma      = 0.1
		seed       = 42
		nSteps     = 10
	)

	b, err := NewBLRTS(nArms, contextDim, alpha, sigma, seed)
	if err != nil {
		t.Fatalf("Failed to create BLRTS: %v", err)
	}

	// True thetas for simulation
	trueThetas := []mat.Vector{
		mat.NewVecDense(contextDim, []float64{1.0, 2.0}),
		mat.NewVecDense(contextDim, []float64{0.5, 1.5}),
		mat.NewVecDense(contextDim, []float64{2.0, 0.5}),
	}

	rng := rand.New(rand.NewSource(123)) // Separate RNG for simulation

	for step := range nSteps {
		// Generate random context
		contextData := make([]float64, contextDim)
		for i := range contextData {
			contextData[i] = rng.NormFloat64()
		}
		context := mat.NewVecDense(contextDim, contextData)

		// Select arm
		arm, err := b.SelectAction(context)
		if err != nil {
			t.Fatalf("SelectAction failed: %v", err)
		}
		if arm < 0 || arm >= nArms {
			t.Errorf("Invalid arm selected: %d", arm)
		}

		// Simulate reward
		trueReward := mat.Dot(trueThetas[arm], context)
		noise := rng.NormFloat64() * sigma
		observedReward := trueReward + noise

		// Update
		b.Update(arm, context, observedReward)

		// Basic check: cov diagonal should have decreased
		if step == 0 {
			for i := range contextDim {
				if b.cov[arm].At(i, i) >= alpha {
					t.Errorf("Cov did not decrease after update for arm %d at diagonal %d: got %f, want < %f", arm, i, b.cov[arm].At(i, i), alpha)
				}
			}
		}
	}
}

func TestBLRTSSaveAndLoad(t *testing.T) {
	const (
		nArms      = 3
		contextDim = 2
		alpha      = 1.0
		sigma      = 0.1
		seed       = 42
		tol        = 1e-10
	)

	b, err := NewBLRTS(nArms, contextDim, alpha, sigma, seed)
	if err != nil {
		t.Fatalf("Failed to create BLRTS: %v", err)
	}

	// Perform multiple updates to change state significantly
	context := mat.NewVecDense(contextDim, []float64{1.0, 1.0})
	for range 10 {
		arm, err := b.SelectAction(context)
		if err != nil {
			t.Fatalf("SelectAction failed: %v", err)
		}
		reward := 1.0
		if arm == 1 {
			reward = 0.5
		}
		b.Update(arm, context, reward)
	}

	// Test Save/Load with io.Reader/Writer interface
	var buf bytes.Buffer
	err = b.Save(&buf)
	if err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Load state
	bRestored, err := Load(&buf, seed)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Compare parameters
	if b.nArms != bRestored.nArms || b.contextDim != bRestored.contextDim ||
		b.alpha != bRestored.alpha || b.sigma != bRestored.sigma {
		t.Errorf("Parameters mismatch after restore")
	}

	// Compare mu and cov for each arm
	for a := range nArms {
		if !mat.EqualApprox(b.mu[a], bRestored.mu[a], tol) {
			t.Errorf("Mu mismatch for arm %d", a)
		}
		if !mat.EqualApprox(b.cov[a], bRestored.cov[a], tol) {
			t.Errorf("Cov mismatch for arm %d", a)
		}
		// Note: chol is recomputed from cov, so we test functional equivalence instead
		// by checking that L*L^T equals the original covariance matrix
		LLt := mat.NewDense(contextDim, contextDim, nil)
		LLt.Mul(bRestored.chol[a], bRestored.chol[a].T())
		if !mat.EqualApprox(LLt, bRestored.cov[a], tol) {
			t.Errorf("Chol functional mismatch for arm %d: L*L^T != cov", a)
		}
	}

	// Check if restored model can make valid selections (Thompson sampling is stochastic)
	context = mat.NewVecDense(contextDim, []float64{0.5, 0.5})
	armOriginal, err := b.SelectAction(context)
	if err != nil {
		t.Fatalf("SelectAction failed on original: %v", err)
	}
	armRestored, err := bRestored.SelectAction(context)
	if err != nil {
		t.Fatalf("SelectAction failed on restored: %v", err)
	}

	// Verify both selections are valid arms (Thompson sampling may differ due to RNG)
	if armOriginal < 0 || armOriginal >= nArms {
		t.Errorf("Invalid arm selection from original model: %d", armOriginal)
	}
	if armRestored < 0 || armRestored >= nArms {
		t.Errorf("Invalid arm selection from restored model: %d", armRestored)
	}
}

func TestBLRTSResetArm(t *testing.T) {
	const (
		nArms      = 3
		contextDim = 2
		alpha      = 1.0
		sigma      = 0.1
	)

	// Create a bandit and train one arm
	b, err := NewBLRTS(nArms, contextDim, alpha, sigma, 42)
	if err != nil {
		t.Fatalf("Failed to create BLRTS: %v", err)
	}
	context := mat.NewVecDense(contextDim, []float64{1.0, 0.5})

	// Update arm 1 multiple times
	for range 10 {
		b.Update(1, context, 1.0)
	}

	// Check that arm 1 has been updated (mu should not be zero)
	muNormBefore := mat.Norm(b.mu[1], 2)
	if muNormBefore < 1e-6 {
		t.Error("Expected arm 1 to be updated, but mu is still ~zero")
	}

	// Reset arm 1
	b.ResetArm(1)

	// Check that arm 1 is back to prior state
	muNormAfter := mat.Norm(b.mu[1], 2)
	if muNormAfter > 1e-10 {
		t.Errorf("Expected arm 1 mu to be zero after reset, got norm: %f", muNormAfter)
	}

	// Check covariance is back to alpha*I
	for i := range contextDim {
		for j := range contextDim {
			expected := 0.0
			if i == j {
				expected = alpha
			}
			if math.Abs(b.cov[1].At(i, j)-expected) > 1e-10 {
				t.Errorf("Cov[1][%d,%d] = %f, expected %f", i, j, b.cov[1].At(i, j), expected)
			}
		}
	}
}

func TestBLRTSEdgeCases(t *testing.T) {
	const (
		alpha = 1.0
		sigma = 0.1
		seed  = 42
	)

	// Test with nArms=1, contextDim=1
	b, err := NewBLRTS(1, 1, alpha, sigma, seed)
	if err != nil {
		t.Fatalf("Failed to create BLRTS: %v", err)
	}

	context := mat.NewVecDense(1, []float64{1.0})
	arm, err := b.SelectAction(context)
	if err != nil {
		t.Fatalf("SelectAction failed: %v", err)
	}
	if arm != 0 {
		t.Errorf("Expected arm 0, got %d", arm)
	}

	if err := b.Update(0, context, 1.0); err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	// Check if denom handles small values
	b.sigma = 1e-6 // Small sigma to force large beta
	if err := b.Update(0, context, 1.0); err != nil {
		t.Fatalf("Update failed: %v", err)
	} // Should not panic

	// Invalid context dimension
	invalidContext := mat.NewVecDense(2, []float64{1.0, 1.0})
	_, err = b.SelectAction(invalidContext)
	if err == nil {
		t.Errorf("Expected error on invalid context dimension")
	}
}

// TestNumericalStability tests that the algorithm remains stable after many updates
func TestNumericalStability(t *testing.T) {
	nArms := 3
	contextDim := 4
	alpha := 1.0
	sigma := 0.1
	seed := int64(42)

	b, err := NewBLRTS(nArms, contextDim, alpha, sigma, seed)
	if err != nil {
		t.Fatalf("Failed to create BLRTS: %v", err)
	}
	context := mat.NewVecDense(contextDim, []float64{1.0, 0.5, -0.3, 0.8})

	// Perform many updates to stress-test numerical stability
	for i := range 100000 {
		arm := i % nArms
		reward := 0.5 + 0.3*float64(arm) + 0.1*rand.NormFloat64()
		b.Update(arm, context, reward)
	}

	// Test max|Σ - LL^T| < 1e-6 for numerical stability after 10k steps
	for a := range nArms {
		// Get Cholesky factorization
		L, err := b.safeChol(b.cov[a])
		if err != nil {
			t.Errorf("Cholesky factorization failed for arm %d: %v", a, err)
			continue
		}

		// Compute L*L^T
		LLt := mat.NewDense(contextDim, contextDim, nil)
		LLt.Mul(L, L.T())

		// Compute max|Σ - LL^T|
		maxDiff := 0.0
		for i := 0; i < contextDim; i++ {
			for j := 0; j < contextDim; j++ {
				diff := math.Abs(b.cov[a].At(i, j) - LLt.At(i, j))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
		}

		// Check numerical stability criterion
		if maxDiff >= 1e-6 {
			t.Errorf("Numerical stability test failed for arm %d: max|Σ - LL^T| = %e >= 1e-6", a, maxDiff)
		}

		// Check that mean updates are finite
		for i := range contextDim {
			val := b.mu[a].AtVec(i)
			if math.IsInf(val, 0) || math.IsNaN(val) {
				t.Errorf("Non-finite mean value for arm %d, component %d: %f", a, i, val)
			}
		}
	}

	// Test that SelectAction still works correctly
	arm, err := b.SelectAction(context)
	if err != nil {
		t.Fatalf("SelectAction failed: %v", err)
	}
	if arm < 0 || arm >= nArms {
		t.Errorf("Invalid arm selection after stress test: %d", arm)
	}

	// Test that Update still works
	b.Update(0, context, 1.0)
}
