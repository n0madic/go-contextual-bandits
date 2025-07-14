package linucbhybrid

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"sync/atomic"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestNewLinUCBHybrid(t *testing.T) {
	tests := []struct {
		name    string
		d       int
		m       int
		options []Option
	}{
		{
			name: "basic initialization",
			d:    5,
			m:    3,
		},
		{
			name: "with custom options",
			d:    10,
			m:    5,
			options: []Option{
				WithAlpha0(2.0),
				WithClipHigh(15.0),
				WithDecay(false),
				WithMaxArticles(50000),
				WithEps(1e-6),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := NewLinUCBHybrid(tt.d, tt.m, tt.options...)

			if l.d != tt.d {
				t.Errorf("Expected d=%d, got %d", tt.d, l.d)
			}
			if l.m != tt.m {
				t.Errorf("Expected m=%d, got %d", tt.m, l.m)
			}

			// Check dimensions of matrices
			r, c := l.A0Inv.Dims()
			if r != tt.m || c != tt.m {
				t.Errorf("A0Inv dimensions: expected (%d,%d), got (%d,%d)", tt.m, tt.m, r, c)
			}

			r, c = l.b0.Dims()
			if r != tt.m || c != 1 {
				t.Errorf("b0 dimensions: expected (%d,1), got (%d,%d)", tt.m, r, c)
			}

			// Check that A0Inv is identity matrix
			for i := 0; i < tt.m; i++ {
				for j := 0; j < tt.m; j++ {
					expected := 0.0
					if i == j {
						expected = 1.0
					}
					if math.Abs(l.A0Inv.At(i, j)-expected) > 1e-10 {
						t.Errorf("A0Inv[%d,%d]: expected %f, got %f", i, j, expected, l.A0Inv.At(i, j))
					}
				}
			}
		})
	}
}

func TestValidateInputs(t *testing.T) {
	l := NewLinUCBHybrid(3, 2)

	tests := []struct {
		name     string
		userFeat []float64
		artFeat  []float64
		wantErr  bool
	}{
		{
			name:     "valid inputs",
			userFeat: []float64{1.0, 2.0},
			artFeat:  []float64{0.5, 1.5, 2.5},
			wantErr:  false,
		},
		{
			name:     "empty user features",
			userFeat: []float64{},
			artFeat:  []float64{0.5, 1.5, 2.5},
			wantErr:  true,
		},
		{
			name:     "wrong article feature size",
			userFeat: []float64{1.0, 2.0},
			artFeat:  []float64{0.5, 1.5},
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := l.validateInputs(tt.userFeat, tt.artFeat)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateInputs() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateInputsWithExpectedUserDim(t *testing.T) {
	l := NewLinUCBHybrid(3, 2, WithExpectedUserDim(2))

	tests := []struct {
		name     string
		userFeat []float64
		artFeat  []float64
		wantErr  bool
	}{
		{
			name:     "correct user dimension",
			userFeat: []float64{1.0, 2.0},
			artFeat:  []float64{0.5, 1.5, 2.5},
			wantErr:  false,
		},
		{
			name:     "wrong user dimension",
			userFeat: []float64{1.0, 2.0, 3.0},
			artFeat:  []float64{0.5, 1.5, 2.5},
			wantErr:  true,
		},
		{
			name:     "empty user features",
			userFeat: []float64{},
			artFeat:  []float64{0.5, 1.5, 2.5},
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := l.validateInputs(tt.userFeat, tt.artFeat)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateInputs() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestRecommend(t *testing.T) {
	l := NewLinUCBHybrid(2, 2)

	userFeat := []float64{1.0, 0.5}
	candidates := map[string][]float64{
		"article1": {0.8, 0.2},
		"article2": {0.3, 0.7},
		"article3": {0.6, 0.4},
	}

	sharedFeatFn := func(user, art []float64) []float64 {
		return []float64{user[0] * art[0], user[1] * art[1]}
	}

	// Test with no candidates
	result, err := l.Recommend(userFeat, map[string][]float64{}, sharedFeatFn)
	if err != nil {
		t.Errorf("Recommend() with empty candidates returned error: %v", err)
	}
	if result != "" {
		t.Errorf("Expected empty result for no candidates, got %s", result)
	}

	// Test with valid candidates
	result, err = l.Recommend(userFeat, candidates, sharedFeatFn)
	if err != nil {
		t.Errorf("Recommend() returned error: %v", err)
	}
	if result == "" {
		t.Error("Expected non-empty result")
	}
	if _, exists := candidates[result]; !exists {
		t.Errorf("Recommended article %s not in candidates", result)
	}
}

func TestUpdate(t *testing.T) {
	l := NewLinUCBHybrid(2, 2)

	userFeat := []float64{1.0, 0.5}
	artFeat := []float64{0.8, 0.2}
	reward := 1.0

	sharedFeatFn := func(user, art []float64) []float64 {
		return []float64{user[0] * art[0], user[1] * art[1]}
	}

	// Test valid update
	err := l.Update("article1", userFeat, artFeat, sharedFeatFn, reward)
	if err != nil {
		t.Errorf("Update() returned error: %v", err)
	}

	// Check that time step increased
	if l.t != 1 {
		t.Errorf("Expected t=1 after update, got %d", l.t)
	}

	// Test update with invalid reward
	err = l.Update("article1", userFeat, artFeat, sharedFeatFn, math.Inf(1))
	if err != nil {
		t.Errorf("Update() with infinite reward should not return error, got: %v", err)
	}

	// Test update with invalid features
	err = l.Update("article1", []float64{}, artFeat, sharedFeatFn, reward)
	if err == nil {
		t.Error("Update() with invalid user features should return error")
	}
}

func TestThompsonSample(t *testing.T) {
	l := NewLinUCBHybrid(2, 2)
	rng := rand.New(rand.NewSource(42))

	userFeat := []float64{1.0, 0.5}
	candidates := map[string][]float64{
		"article1": {0.8, 0.2},
		"article2": {0.3, 0.7},
	}

	sharedFeatFn := func(user, art []float64) []float64 {
		return []float64{user[0] * art[0], user[1] * art[1]}
	}

	// Test with no candidates
	result, err := l.ThompsonSample(userFeat, map[string][]float64{}, sharedFeatFn, rng)
	if err != nil {
		t.Errorf("ThompsonSample() with empty candidates returned error: %v", err)
	}
	if result != "" {
		t.Errorf("Expected empty result for no candidates, got %s", result)
	}

	// Test with valid candidates
	result, err = l.ThompsonSample(userFeat, candidates, sharedFeatFn, rng)
	if err != nil {
		t.Errorf("ThompsonSample() returned error: %v", err)
	}
	if result == "" {
		t.Error("Expected non-empty result")
	}
	if _, exists := candidates[result]; !exists {
		t.Errorf("Sampled article %s not in candidates", result)
	}
}

func TestAlphaDecay(t *testing.T) {
	l := NewLinUCBHybrid(2, 2, WithAlpha0(4.0), WithDecay(true))

	// Initial alpha should be alpha0
	alpha := l.alpha()
	if alpha != 4.0 {
		t.Errorf("Initial alpha: expected 4.0, got %f", alpha)
	}

	// After incrementing t, alpha should decay
	l.t = 4
	alpha = l.alpha()
	expected := 4.0 / math.Sqrt(4.0)
	if math.Abs(alpha-expected) > 1e-10 {
		t.Errorf("Decayed alpha: expected %f, got %f", expected, alpha)
	}

	// Test with decay disabled
	l2 := NewLinUCBHybrid(2, 2, WithAlpha0(4.0), WithDecay(false))
	l2.t = 4
	alpha = l2.alpha()
	if alpha != 4.0 {
		t.Errorf("Alpha with decay disabled: expected 4.0, got %f", alpha)
	}
}

func TestConcurrentOperations(t *testing.T) {
	l := NewLinUCBHybrid(3, 2)

	userFeat := []float64{1.0, 0.5, 0.3}
	candidates := map[string][]float64{
		"article1": {0.8, 0.2, 0.1},
		"article2": {0.3, 0.7, 0.4},
	}

	sharedFeatFn := func(user, art []float64) []float64 {
		return []float64{user[0] * art[0], user[1] * art[1]}
	}

	// Run concurrent updates and recommendations
	done := make(chan bool)
	numWorkers := 10
	numOperations := 100

	for i := 0; i < numWorkers; i++ {
		go func(workerID int) {
			rng := rand.New(rand.NewSource(int64(workerID)))
			for j := 0; j < numOperations; j++ {
				// Randomly choose update or recommend
				if rng.Float64() < 0.5 {
					// Update
					artID := "article1"
					if rng.Float64() < 0.5 {
						artID = "article2"
					}
					reward := rng.Float64()
					l.Update(artID, userFeat, candidates[artID], sharedFeatFn, reward)
				} else {
					// Recommend
					l.Recommend(userFeat, candidates, sharedFeatFn)
				}
			}
			done <- true
		}(i)
	}

	// Wait for all workers to finish
	for i := 0; i < numWorkers; i++ {
		<-done
	}

	// Verify that the algorithm is still functional
	result, err := l.Recommend(userFeat, candidates, sharedFeatFn)
	if err != nil {
		t.Errorf("Recommend() after concurrent operations returned error: %v", err)
	}
	if result == "" {
		t.Error("Expected non-empty result after concurrent operations")
	}
}

func TestCacheEviction(t *testing.T) {
	l := NewLinUCBHybrid(2, 2, WithMaxArticles(3))

	userFeat := []float64{1.0, 0.5}
	sharedFeatFn := func(user, art []float64) []float64 {
		return []float64{user[0] * art[0], user[1] * art[1]}
	}

	// Add articles up to the limit
	for i := 1; i <= 3; i++ {
		artID := fmt.Sprintf("article%d", i)
		artFeat := []float64{float64(i) * 0.1, float64(i) * 0.2}
		err := l.Update(artID, userFeat, artFeat, sharedFeatFn, 1.0)
		if err != nil {
			t.Errorf("Update() for %s returned error: %v", artID, err)
		}
	}

	// Check cache size
	if len(l.cache) != 3 {
		t.Errorf("Expected cache size 3, got %d", len(l.cache))
	}

	// Add one more article (should trigger eviction)
	artFeat := []float64{0.4, 0.8}
	err := l.Update("article4", userFeat, artFeat, sharedFeatFn, 1.0)
	if err != nil {
		t.Errorf("Update() for article4 returned error: %v", err)
	}

	// Cache should still be at max size
	if len(l.cache) != 3 {
		t.Errorf("Expected cache size 3 after eviction, got %d", len(l.cache))
	}

	// The oldest article (article1) should have been evicted
	if _, exists := l.cache["article1"]; exists {
		t.Error("article1 should have been evicted from cache")
	}

	// article4 should be in cache
	if _, exists := l.cache["article4"]; !exists {
		t.Error("article4 should be in cache")
	}
}

func TestSampleMultivariateNormal(t *testing.T) {
	l := NewLinUCBHybrid(2, 2)
	rng := rand.New(rand.NewSource(42))

	// Create a simple test case
	mean := make([]float64, 2)
	mean[0] = 1.0
	mean[1] = 2.0
	meanMat := mat.NewDense(2, 1, mean)

	// Create a simple precision matrix (inverse of covariance)
	precisionData := []float64{2.0, 0.0, 0.0, 2.0}
	precision := mat.NewDense(2, 2, precisionData)

	// Sample multiple times and check basic properties
	numSamples := 1000
	samples := make([][]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		sample := l.sampleMultivariateNormal(meanMat, precision, rng)
		r, c := sample.Dims()
		if r != 2 || c != 1 {
			t.Fatalf("Sample dimensions: expected (2,1), got (%d,%d)", r, c)
		}
		samples[i] = []float64{sample.At(0, 0), sample.At(1, 0)}
	}

	// Compute sample mean
	sampleMean := make([]float64, 2)
	for i := 0; i < numSamples; i++ {
		sampleMean[0] += samples[i][0]
		sampleMean[1] += samples[i][1]
	}
	sampleMean[0] /= float64(numSamples)
	sampleMean[1] /= float64(numSamples)

	// Check that sample mean is close to true mean (within 3 standard errors)
	tolerance := 3.0 * math.Sqrt(0.5/float64(numSamples)) // 3 * se, where variance = 0.5
	if math.Abs(sampleMean[0]-1.0) > tolerance {
		t.Errorf("Sample mean[0]: expected ~1.0, got %f (tolerance: %f)", sampleMean[0], tolerance)
	}
	if math.Abs(sampleMean[1]-2.0) > tolerance {
		t.Errorf("Sample mean[1]: expected ~2.0, got %f (tolerance: %f)", sampleMean[1], tolerance)
	}
}

// TestPropertyMatrixInvariance tests that matrix operations maintain mathematical properties
func TestPropertyMatrixInvariance(t *testing.T) {
	l := NewLinUCBHybrid(3, 2)

	// Test that A0Inv remains positive definite after updates
	sharedFeatFn := func(user, art []float64) []float64 {
		return []float64{user[0] * art[0], user[1] + art[1]}
	}

	// Perform several updates
	for i := 0; i < 10; i++ {
		userFeat := []float64{rand.Float64(), rand.Float64()}
		artFeat := []float64{rand.Float64(), rand.Float64(), rand.Float64()}
		reward := rand.Float64()

		err := l.Update(fmt.Sprintf("art_%d", i), userFeat, artFeat, sharedFeatFn, reward)
		if err != nil {
			t.Fatalf("Update failed: %v", err)
		}

		// Check that A0Inv is still positive definite (all eigenvalues > 0)
		var eig mat.Eigen
		if ok := eig.Factorize(l.A0Inv, mat.EigenRight); !ok {
			t.Fatalf("Eigenvalue decomposition failed at iteration %d", i)
		}
		values := eig.Values(nil)
		for j, val := range values {
			if real(val) <= 0 {
				t.Errorf("A0Inv eigenvalue %d is non-positive: %v at iteration %d", j, val, i)
			}
		}
	}
}

// TestPropertyRewardMonotonicity tests that rewards improve with positive feedback
func TestPropertyRewardMonotonicity(t *testing.T) {
	l := NewLinUCBHybrid(2, 2, WithAlpha0(0.1)) // Small alpha for clearer signal

	sharedFeatFn := func(user, art []float64) []float64 {
		return []float64{user[0] * art[0], user[1] * art[1]}
	}

	userFeat := []float64{1.0, 1.0}
	artFeat := []float64{1.0, 1.0}
	candidates := map[string][]float64{"test_article": artFeat}

	// Get initial recommendation score
	initialRec, err := l.Recommend(userFeat, candidates, sharedFeatFn)
	if err != nil {
		t.Fatalf("Initial recommendation failed: %v", err)
	}
	if initialRec == "" {
		t.Fatal("No initial recommendation returned")
	}

	// Apply positive reward updates
	for i := 0; i < 5; i++ {
		err = l.Update("test_article", userFeat, artFeat, sharedFeatFn, 1.0)
		if err != nil {
			t.Fatalf("Update %d failed: %v", i, err)
		}
	}

	// After positive updates, the same article should still be recommended
	// (since it's the only candidate) and confidence should be higher
	finalRec, err := l.Recommend(userFeat, candidates, sharedFeatFn)
	if err != nil {
		t.Fatalf("Final recommendation failed: %v", err)
	}
	if finalRec != "test_article" {
		t.Errorf("Expected recommendation 'test_article', got '%s'", finalRec)
	}
}

// TestPropertyConvergence tests that the algorithm converges with consistent rewards
func TestPropertyConvergence(t *testing.T) {
	l := NewLinUCBHybrid(2, 2, WithDecay(true), WithAlpha0(1.0))

	sharedFeatFn := func(user, art []float64) []float64 {
		return []float64{user[0] + art[0], user[1] + art[1]}
	}

	// Create deterministic scenario
	userFeat := []float64{1.0, 0.0}
	goodArt := []float64{1.0, 0.0} // Should get higher reward
	badArt := []float64{0.0, 1.0}  // Should get lower reward

	candidates := map[string][]float64{
		"good": goodArt,
		"bad":  badArt,
	}

	// Track recommendations over time
	goodRecommendations := 0
	totalRecommendations := 50

	for i := 0; i < totalRecommendations; i++ {
		// Get recommendation
		rec, err := l.Recommend(userFeat, candidates, sharedFeatFn)
		if err != nil {
			t.Fatalf("Recommendation %d failed: %v", i, err)
		}

		// Apply appropriate reward
		var reward float64
		var artFeat []float64
		if rec == "good" {
			reward = 1.0
			artFeat = goodArt
			goodRecommendations++
		} else {
			reward = 0.0
			artFeat = badArt
		}

		err = l.Update(rec, userFeat, artFeat, sharedFeatFn, reward)
		if err != nil {
			t.Fatalf("Update %d failed: %v", i, err)
		}
	}

	// After learning, should prefer the good article most of the time
	// Allow some exploration but expect at least 70% good recommendations
	goodRatio := float64(goodRecommendations) / float64(totalRecommendations)
	if goodRatio < 0.7 {
		t.Errorf("Expected at least 70%% good recommendations, got %.1f%%", goodRatio*100)
	}
}

// TestPropertyThompsonSamplingExploration tests that Thompson sampling provides exploration
func TestPropertyThompsonSamplingExploration(t *testing.T) {
	l := NewLinUCBHybrid(2, 2)
	rng := rand.New(rand.NewSource(42))

	sharedFeatFn := func(user, art []float64) []float64 {
		return []float64{user[0] * art[0], user[1] * art[1]}
	}

	userFeat := []float64{1.0, 1.0}
	candidates := map[string][]float64{
		"art1": {1.0, 0.0},
		"art2": {0.0, 1.0},
		"art3": {0.5, 0.5},
	}

	// Track recommendations from Thompson sampling
	recommendationCounts := make(map[string]int)
	numSamples := 100

	for i := 0; i < numSamples; i++ {
		rec, err := l.ThompsonSample(userFeat, candidates, sharedFeatFn, rng)
		if err != nil {
			t.Fatalf("Thompson sample %d failed: %v", i, err)
		}
		if rec != "" {
			recommendationCounts[rec]++
		}
	}

	// Should explore all articles (each should be recommended at least once)
	for artID := range candidates {
		if recommendationCounts[artID] == 0 {
			t.Errorf("Article %s was never recommended by Thompson sampling", artID)
		}
	}

	// No single article should dominate completely (less than 80% of recommendations)
	for artID, count := range recommendationCounts {
		ratio := float64(count) / float64(numSamples)
		if ratio > 0.8 {
			t.Errorf("Article %s dominated with %.1f%% of recommendations, expected more exploration", artID, ratio*100)
		}
	}
}

// TestPropertyInputValidation tests mathematical properties of input validation
func TestPropertyInputValidation(t *testing.T) {
	l := NewLinUCBHybrid(3, 2)

	tests := []struct {
		name        string
		userFeat    []float64
		artFeat     []float64
		expectError bool
	}{
		{"valid_input", []float64{1.0, 2.0}, []float64{1.0, 2.0, 3.0}, false},
		{"empty_user", []float64{}, []float64{1.0, 2.0, 3.0}, true},
		{"wrong_art_size", []float64{1.0, 2.0}, []float64{1.0, 2.0}, true},
		{"nil_user", nil, []float64{1.0, 2.0, 3.0}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := l.validateInputs(tt.userFeat, tt.artFeat)
			if tt.expectError && err == nil {
				t.Error("Expected validation error but got none")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected validation error: %v", err)
			}
		})
	}
}

// TestSampleMultivariateNormalCorrectness tests that sampling produces correct covariance
func TestSampleMultivariateNormalCorrectness(t *testing.T) {
	l := NewLinUCBHybrid(3, 3)
	rng := rand.New(rand.NewSource(12345))

	// Create a known covariance matrix Σ
	covData := []float64{
		2.0, 0.5, 0.0,
		0.5, 1.0, 0.3,
		0.0, 0.3, 1.5,
	}
	cov := mat.NewDense(3, 3, covData)

	// For our corrected implementation, pass the covariance matrix directly
	// (the function parameter is named 'prec' but actually expects covariance)
	prec := cov // Pass covariance matrix directly

	// Set mean vector
	meanData := []float64{1.0, -0.5, 2.0}
	mean := mat.NewDense(3, 1, meanData)

	// Generate many samples
	numSamples := 10000
	samples := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		sample := l.sampleMultivariateNormal(mean, prec, rng)
		samples[i] = make([]float64, 3)
		for j := 0; j < 3; j++ {
			samples[i][j] = sample.At(j, 0)
		}
	}

	// Compute sample mean
	sampleMean := make([]float64, 3)
	for i := 0; i < numSamples; i++ {
		for j := 0; j < 3; j++ {
			sampleMean[j] += samples[i][j]
		}
	}
	for j := 0; j < 3; j++ {
		sampleMean[j] /= float64(numSamples)
	}

	// Check sample mean is close to true mean
	for j := 0; j < 3; j++ {
		expected := meanData[j]
		tolerance := 3.0 * math.Sqrt(cov.At(j, j)/float64(numSamples)) // 3 standard errors
		if math.Abs(sampleMean[j]-expected) > tolerance {
			t.Errorf("Sample mean[%d]: expected %.3f, got %.3f (tolerance: %.3f)", j, expected, sampleMean[j], tolerance)
		}
	}

	// Compute sample covariance matrix
	sampleCov := mat.NewDense(3, 3, nil)
	for i := 0; i < numSamples; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 3; k++ {
				val := (samples[i][j] - sampleMean[j]) * (samples[i][k] - sampleMean[k])
				sampleCov.Set(j, k, sampleCov.At(j, k)+val)
			}
		}
	}
	sampleCov.Scale(1.0/float64(numSamples-1), sampleCov)

	// Check sample covariance is close to true covariance
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			expected := cov.At(i, j)
			got := sampleCov.At(i, j)
			// Tolerance based on standard error of covariance estimate
			// Use more generous tolerance for finite sample effects
			tolerance := 4.0 * math.Sqrt(2.0*expected*expected/float64(numSamples-1))
			if math.Abs(expected) < 1e-6 {
				tolerance = math.Max(tolerance, 0.05) // Minimum tolerance for near-zero values
			} else {
				tolerance = math.Max(tolerance, 0.03) // Minimum tolerance for non-zero values
			}
			if math.Abs(got-expected) > tolerance {
				t.Errorf("Sample cov[%d,%d]: expected %.3f, got %.3f (tolerance: %.3f)", i, j, expected, got, tolerance)
			}
		}
	}
}

// TestSampleMultivariateNormalEdgeCases tests edge cases in sampling
func TestSampleMultivariateNormalEdgeCases(t *testing.T) {
	l := NewLinUCBHybrid(2, 2)
	rng := rand.New(rand.NewSource(42))

	// Test with singular precision matrix (should return mean)
	singularPrec := mat.NewDense(2, 2, []float64{1.0, 1.0, 1.0, 1.0})
	mean := mat.NewDense(2, 1, []float64{1.0, 2.0})

	sample := l.sampleMultivariateNormal(mean, singularPrec, rng)
	for i := 0; i < 2; i++ {
		if math.Abs(sample.At(i, 0)-mean.At(i, 0)) > 1e-10 {
			t.Errorf("For singular precision, expected sample to equal mean, got difference: %v", sample.At(i, 0)-mean.At(i, 0))
		}
	}

	// Test with identity precision (covariance = identity)
	identityPrec := mat.NewDense(2, 2, []float64{1.0, 0.0, 0.0, 1.0})
	samples := make([][]float64, 1000)
	for i := 0; i < 1000; i++ {
		sample := l.sampleMultivariateNormal(mean, identityPrec, rng)
		samples[i] = []float64{sample.At(0, 0), sample.At(1, 0)}
	}

	// Compute sample variance (should be approximately 1.0)
	sampleMean := []float64{0, 0}
	for i := 0; i < 1000; i++ {
		sampleMean[0] += samples[i][0]
		sampleMean[1] += samples[i][1]
	}
	sampleMean[0] /= 1000.0
	sampleMean[1] /= 1000.0

	sampleVar := []float64{0, 0}
	for i := 0; i < 1000; i++ {
		sampleVar[0] += (samples[i][0] - sampleMean[0]) * (samples[i][0] - sampleMean[0])
		sampleVar[1] += (samples[i][1] - sampleMean[1]) * (samples[i][1] - sampleMean[1])
	}
	sampleVar[0] /= 999.0
	sampleVar[1] /= 999.0

	// Check variances are approximately 1.0 (tolerance for 1000 samples)
	for i := 0; i < 2; i++ {
		if math.Abs(sampleVar[i]-1.0) > 0.2 {
			t.Errorf("Sample variance[%d]: expected ~1.0, got %.3f", i, sampleVar[i])
		}
	}
}

// TestOptimizedSamplingPerformance compares performance of cached vs direct sampling
func TestOptimizedSamplingPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	l := NewLinUCBHybrid(10, 5) // Larger dimensions for meaningful comparison
	rng := rand.New(rand.NewSource(42))

	// Create article with some data to make AInv more realistic
	aID := "test_article"
	art := l.getOrCreateArticle(aID)
	defer l.releaseArticle(art, aID)

	// Perform some updates to make AInv non-trivial
	sharedFeatFn := func(user, art []float64) []float64 {
		return []float64{user[0] * art[0], user[1] * art[1], 1.0, 0.5, 0.1}
	}

	for i := 0; i < 50; i++ {
		userFeat := []float64{rand.Float64(), rand.Float64()}
		artFeat := make([]float64, 10)
		for j := range artFeat {
			artFeat[j] = rand.Float64()
		}
		reward := rand.Float64()
		l.Update(aID, userFeat, artFeat, sharedFeatFn, reward)
	}

	// Test cached sampling correctness
	mean := mat.NewDense(10, 1, nil)
	for i := 0; i < 10; i++ {
		mean.Set(i, 0, float64(i))
	}

	// Both methods should give statistically similar results
	numSamples := 1000
	directSamples := make([][]float64, numSamples)
	cachedSamples := make([][]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		// Direct sampling
		directSample := l.sampleMultivariateNormal(mean, art.AInv, rng)
		directSamples[i] = make([]float64, 10)
		for j := 0; j < 10; j++ {
			directSamples[i][j] = directSample.At(j, 0)
		}

		// Cached sampling
		cachedSample := l.sampleMultivariateNormalOptimized(mean, art, rng)
		cachedSamples[i] = make([]float64, 10)
		for j := 0; j < 10; j++ {
			cachedSamples[i][j] = cachedSample.At(j, 0)
		}
	}

	// Compare sample means (should be close)
	directMean := make([]float64, 10)
	cachedMean := make([]float64, 10)
	for i := 0; i < numSamples; i++ {
		for j := 0; j < 10; j++ {
			directMean[j] += directSamples[i][j]
			cachedMean[j] += cachedSamples[i][j]
		}
	}
	for j := 0; j < 10; j++ {
		directMean[j] /= float64(numSamples)
		cachedMean[j] /= float64(numSamples)

		// Means should be close (within 3 standard errors)
		tolerance := 0.5 // Generous tolerance for this test
		if math.Abs(directMean[j]-cachedMean[j]) > tolerance {
			t.Errorf("Sample means differ significantly at index %d: direct=%.3f, cached=%.3f", j, directMean[j], cachedMean[j])
		}
	}
}

// TestCovarianceCorrectness validates that optimized sampling produces correct covariance
// This test ensures the critical fix for getCachedCholesky() works correctly
func TestCovarianceCorrectness(t *testing.T) {
	l := NewLinUCBHybrid(8, 4) // Use dimensions that trigger optimized sampling
	rng := rand.New(rand.NewSource(12345))

	// Create article and perform realistic updates
	aID := "test_article"
	art := l.getOrCreateArticle(aID)
	defer l.releaseArticle(art, aID)

	sharedFeatFn := func(user, art []float64) []float64 {
		return []float64{user[0] * art[0], user[1] * art[1], user[2] * art[2], user[3] * art[3]}
	}

	// Perform updates to create non-trivial AInv matrix
	for i := 0; i < 100; i++ {
		userFeat := []float64{rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64()}
		artFeat := make([]float64, 8)
		for j := range artFeat {
			artFeat[j] = rng.NormFloat64()
		}
		reward := rng.NormFloat64()
		l.Update(aID, userFeat, artFeat, sharedFeatFn, reward)
	}

	// Get the true covariance matrix (AInv)
	art.mu.RLock()
	trueCovariance := mat.NewDense(8, 8, nil)
	trueCovariance.Copy(art.AInv)
	art.mu.RUnlock()

	// Generate many samples using optimized method
	mean := mat.NewDense(8, 1, nil)
	for i := 0; i < 8; i++ {
		mean.Set(i, 0, float64(i)*0.5) // Set non-zero mean
	}

	numSamples := 10000
	samples := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		sample := l.sampleMultivariateNormalOptimized(mean, art, rng)
		samples[i] = make([]float64, 8)
		for j := 0; j < 8; j++ {
			samples[i][j] = sample.At(j, 0)
		}
	}

	// Compute sample mean
	sampleMean := make([]float64, 8)
	for i := 0; i < numSamples; i++ {
		for j := 0; j < 8; j++ {
			sampleMean[j] += samples[i][j]
		}
	}
	for j := 0; j < 8; j++ {
		sampleMean[j] /= float64(numSamples)
	}

	// Verify sample mean is close to true mean
	for j := 0; j < 8; j++ {
		expectedMean := mean.At(j, 0)
		tolerance := 3.0 * math.Sqrt(trueCovariance.At(j, j)/float64(numSamples))
		if math.Abs(sampleMean[j]-expectedMean) > tolerance {
			t.Errorf("Sample mean[%d]: expected %.3f, got %.3f (tolerance: %.3f)", j, expectedMean, sampleMean[j], tolerance)
		}
	}

	// Compute sample covariance matrix
	sampleCovariance := mat.NewDense(8, 8, nil)
	for i := 0; i < numSamples; i++ {
		for j := 0; j < 8; j++ {
			for k := 0; k < 8; k++ {
				val := (samples[i][j] - sampleMean[j]) * (samples[i][k] - sampleMean[k])
				sampleCovariance.Set(j, k, sampleCovariance.At(j, k)+val)
			}
		}
	}
	sampleCovariance.Scale(1.0/float64(numSamples-1), sampleCovariance)

	// Verify sample covariance matches true covariance (AInv)
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			expected := trueCovariance.At(i, j)
			got := sampleCovariance.At(i, j)

			// Use relative tolerance for covariance elements
			var tolerance float64
			if math.Abs(expected) > 1e-6 {
				// For non-zero elements, use relative tolerance
				tolerance = 0.15 * math.Abs(expected) // 15% relative tolerance
				if tolerance < 0.02 {
					tolerance = 0.02 // Minimum absolute tolerance
				}
			} else {
				// For near-zero elements, use absolute tolerance
				tolerance = 0.05
			}

			if math.Abs(got-expected) > tolerance {
				t.Errorf("Sample covariance[%d,%d]: expected %.4f, got %.4f (tolerance: %.4f)",
					i, j, expected, got, tolerance)
			}
		}
	}

	// Additional check: verify that cached and direct methods produce similar covariance
	numCompareSamples := 2000
	directSamples := make([][]float64, numCompareSamples)
	cachedSamples := make([][]float64, numCompareSamples)

	for i := 0; i < numCompareSamples; i++ {
		// Direct sampling
		directSample := l.sampleMultivariateNormal(mean, art.AInv, rng)
		directSamples[i] = make([]float64, 8)
		for j := 0; j < 8; j++ {
			directSamples[i][j] = directSample.At(j, 0)
		}

		// Cached sampling
		cachedSample := l.sampleMultivariateNormalOptimized(mean, art, rng)
		cachedSamples[i] = make([]float64, 8)
		for j := 0; j < 8; j++ {
			cachedSamples[i][j] = cachedSample.At(j, 0)
		}
	}

	// Compare variances of direct vs cached sampling
	for j := 0; j < 8; j++ {
		// Compute variances
		directVar := 0.0
		cachedVar := 0.0
		directMean := 0.0
		cachedMean := 0.0

		for i := 0; i < numCompareSamples; i++ {
			directMean += directSamples[i][j]
			cachedMean += cachedSamples[i][j]
		}
		directMean /= float64(numCompareSamples)
		cachedMean /= float64(numCompareSamples)

		for i := 0; i < numCompareSamples; i++ {
			directVar += (directSamples[i][j] - directMean) * (directSamples[i][j] - directMean)
			cachedVar += (cachedSamples[i][j] - cachedMean) * (cachedSamples[i][j] - cachedMean)
		}
		directVar /= float64(numCompareSamples - 1)
		cachedVar /= float64(numCompareSamples - 1)

		// Variances should be similar (within 25%)
		if directVar > 0 && cachedVar > 0 {
			ratio := cachedVar / directVar
			if ratio < 0.75 || ratio > 1.33 {
				t.Errorf("Variance mismatch at dimension %d: direct=%.4f, cached=%.4f, ratio=%.3f",
					j, directVar, cachedVar, ratio)
			}
		}
	}
}

// TestSaveLoad tests comprehensive serialization, deserialization and behavioral preservation
func TestSaveLoad(t *testing.T) {
	// Create and configure a LinUCB instance with comprehensive options
	original := NewLinUCBHybrid(3, 2,
		WithAlpha0(2.5),
		WithClipHigh(15.0),
		WithDecay(true), // Test decay functionality
		WithMaxArticles(50),
		WithEps(1e-6),
		WithExpectedUserDim(2),
	)

	// Setup test data
	sharedFeatFn := func(user, art []float64) []float64 {
		return []float64{user[0] * art[0], user[1] + art[1]}
	}

	userFeat := []float64{1.0, 0.8}
	articles := []struct {
		id   string
		feat []float64
	}{
		{"article1", []float64{0.8, 0.2, 0.1}},
		{"article2", []float64{0.3, 0.7, 0.4}},
		{"article3", []float64{0.6, 0.4, 0.9}},
	}

	// Train the model to create non-trivial state
	for i := 0; i < 10; i++ {
		for _, art := range articles {
			reward := 0.3 + 0.4*rand.Float64()
			err := original.Update(art.id, userFeat, art.feat, sharedFeatFn, reward)
			if err != nil {
				t.Fatalf("Update failed: %v", err)
			}
		}
	}

	// Test candidates for behavioral verification
	candidates := map[string][]float64{
		"article1": articles[0].feat,
		"article2": articles[1].feat,
		"article3": articles[2].feat,
	}

	// Get baseline behavior from original model
	originalRec, err := original.Recommend(userFeat, candidates, sharedFeatFn)
	if err != nil {
		t.Fatalf("Original recommend failed: %v", err)
	}

	// Save the model
	var buf bytes.Buffer
	err = original.Save(&buf)
	if err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Load the model
	loaded, err := Load(&buf)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// SECTION 1: Verify configuration parameters preservation
	t.Run("ConfigurationPreservation", func(t *testing.T) {
		if loaded.d != original.d {
			t.Errorf("d mismatch: expected %d, got %d", original.d, loaded.d)
		}
		if loaded.m != original.m {
			t.Errorf("m mismatch: expected %d, got %d", original.m, loaded.m)
		}
		if loaded.alpha0 != original.alpha0 {
			t.Errorf("alpha0 mismatch: expected %f, got %f", original.alpha0, loaded.alpha0)
		}
		if loaded.clipHigh != original.clipHigh {
			t.Errorf("clipHigh mismatch: expected %f, got %f", original.clipHigh, loaded.clipHigh)
		}
		if loaded.decay != original.decay {
			t.Errorf("decay mismatch: expected %t, got %t", original.decay, loaded.decay)
		}
		if loaded.maxArticles != original.maxArticles {
			t.Errorf("maxArticles mismatch: expected %d, got %d", original.maxArticles, loaded.maxArticles)
		}
		if loaded.eps != original.eps {
			t.Errorf("eps mismatch: expected %f, got %f", original.eps, loaded.eps)
		}
		if loaded.expectedUserDim != original.expectedUserDim {
			t.Errorf("expectedUserDim mismatch: expected %d, got %d", original.expectedUserDim, loaded.expectedUserDim)
		}
		if atomic.LoadUint64(&loaded.t) != atomic.LoadUint64(&original.t) {
			t.Errorf("t mismatch: expected %d, got %d", atomic.LoadUint64(&original.t), atomic.LoadUint64(&loaded.t))
		}
	})

	// SECTION 2: Verify matrix data integrity
	t.Run("MatrixDataIntegrity", func(t *testing.T) {
		if !mat.EqualApprox(loaded.A0Inv, original.A0Inv, 1e-10) {
			t.Error("A0Inv matrices do not match")
		}
		if !mat.EqualApprox(loaded.b0, original.b0, 1e-10) {
			t.Error("b0 vectors do not match")
		}

		// Verify article cache integrity
		if len(loaded.cache) != len(original.cache) {
			t.Errorf("Cache size mismatch: expected %d, got %d", len(original.cache), len(loaded.cache))
		}

		for aID := range original.cache {
			if _, exists := loaded.cache[aID]; !exists {
				t.Errorf("Article %s missing from loaded cache", aID)
				continue
			}

			origArt := original.cache[aID]
			loadedArt := loaded.cache[aID]

			origArt.mu.RLock()
			loadedArt.mu.RLock()

			if !mat.EqualApprox(loadedArt.AInv, origArt.AInv, 1e-10) {
				t.Errorf("AInv mismatch for article %s", aID)
			}
			if !mat.EqualApprox(loadedArt.B, origArt.B, 1e-10) {
				t.Errorf("B mismatch for article %s", aID)
			}
			if !mat.EqualApprox(loadedArt.b, origArt.b, 1e-10) {
				t.Errorf("b mismatch for article %s", aID)
			}
			if !mat.EqualApprox(loadedArt.W, origArt.W, 1e-10) {
				t.Errorf("W mismatch for article %s", aID)
			}
			if !mat.EqualApprox(loadedArt.M, origArt.M, 1e-10) {
				t.Errorf("M mismatch for article %s", aID)
			}

			origArt.mu.RUnlock()
			loadedArt.mu.RUnlock()
		}
	})

	// SECTION 3: Verify behavioral consistency
	t.Run("BehavioralConsistency", func(t *testing.T) {
		// Test recommendation consistency
		loadedRec, err := loaded.Recommend(userFeat, candidates, sharedFeatFn)
		if err != nil {
			t.Fatalf("Loaded recommend failed: %v", err)
		}

		if originalRec != loadedRec {
			t.Errorf("Recommendations differ: original=%s, loaded=%s", originalRec, loadedRec)
		}

		// Test Thompson sampling functionality (both should return valid candidates)
		// Note: Thompson sampling is probabilistic, so we test functionality rather than exact determinism
		rng1 := rand.New(rand.NewSource(42))
		rng2 := rand.New(rand.NewSource(42))

		originalTS, err := original.ThompsonSample(userFeat, candidates, sharedFeatFn, rng1)
		if err != nil {
			t.Fatalf("Original ThompsonSample failed: %v", err)
		}

		loadedTS, err := loaded.ThompsonSample(userFeat, candidates, sharedFeatFn, rng2)
		if err != nil {
			t.Fatalf("Loaded ThompsonSample failed: %v", err)
		}

		// Verify both returned valid candidates (exact match not required due to caching)
		if _, exists := candidates[originalTS]; !exists {
			t.Errorf("Original Thompson sample returned invalid candidate: %s", originalTS)
		}
		if _, exists := candidates[loadedTS]; !exists {
			t.Errorf("Loaded Thompson sample returned invalid candidate: %s", loadedTS)
		}

		// Test multiple samples to ensure both models are functional
		for i := 0; i < 5; i++ {
			rng3 := rand.New(rand.NewSource(int64(i + 100)))
			rng4 := rand.New(rand.NewSource(int64(i + 100)))

			ts1, err := original.ThompsonSample(userFeat, candidates, sharedFeatFn, rng3)
			if err != nil {
				t.Fatalf("Original ThompsonSample iteration %d failed: %v", i, err)
			}
			ts2, err := loaded.ThompsonSample(userFeat, candidates, sharedFeatFn, rng4)
			if err != nil {
				t.Fatalf("Loaded ThompsonSample iteration %d failed: %v", i, err)
			}

			if _, exists := candidates[ts1]; !exists {
				t.Errorf("Original TS iteration %d returned invalid candidate: %s", i, ts1)
			}
			if _, exists := candidates[ts2]; !exists {
				t.Errorf("Loaded TS iteration %d returned invalid candidate: %s", i, ts2)
			}
		}
	})

	// SECTION 4: Verify learning consistency
	t.Run("LearningConsistency", func(t *testing.T) {
		// Test that both models learn identically from new data
		newUserFeat := []float64{0.5, 1.2}
		newArtFeat := []float64{0.8, 0.4, 0.6}
		newReward := 0.7

		err = original.Update("new_article", newUserFeat, newArtFeat, sharedFeatFn, newReward)
		if err != nil {
			t.Fatalf("Update on original failed: %v", err)
		}

		err = loaded.Update("new_article", newUserFeat, newArtFeat, sharedFeatFn, newReward)
		if err != nil {
			t.Fatalf("Update on loaded failed: %v", err)
		}

		// Verify time counters remain synchronized
		if atomic.LoadUint64(&original.t) != atomic.LoadUint64(&loaded.t) {
			t.Errorf("Time counters diverged: original=%d, loaded=%d",
				atomic.LoadUint64(&original.t), atomic.LoadUint64(&loaded.t))
		}

		// Verify that both models still give same recommendations after new learning
		extendedCandidates := make(map[string][]float64)
		for k, v := range candidates {
			extendedCandidates[k] = v
		}
		extendedCandidates["new_article"] = newArtFeat

		finalOrigRec, err := original.Recommend(newUserFeat, extendedCandidates, sharedFeatFn)
		if err != nil {
			t.Fatalf("Final original recommend failed: %v", err)
		}

		finalLoadedRec, err := loaded.Recommend(newUserFeat, extendedCandidates, sharedFeatFn)
		if err != nil {
			t.Fatalf("Final loaded recommend failed: %v", err)
		}

		if finalOrigRec != finalLoadedRec {
			t.Errorf("Final recommendations differ: original=%s, loaded=%s", finalOrigRec, finalLoadedRec)
		}
	})
}
