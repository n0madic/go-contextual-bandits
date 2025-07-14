package lints

import (
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"gonum.org/v1/gonum/mat"
)

// LinTS implements Linear Thompson Sampling for contextual bandits.
// This is a port of the Python implementation with the following features:
// - O(d²) Sherman-Morrison updates with rank-1 Cholesky updates
// - L2-normalized contexts for numerical stability
// - Optional bias term with proper 1/√d scaling
// - Batch scoring with shared θ sample
// - Numerical stability with periodic maintenance
// - Real-valued rewards (no artificial binarization)
// - Thread-safe operations for concurrent access
type LinTS struct {
	dFeatures       int     // feature dimension (without bias)
	d               int     // actual dimension (with bias if enabled)
	lambda          float64 // regularization parameter
	sigma2          float64 // noise variance for Thompson sampling
	maintenanceFreq int     // perform numerical maintenance every N updates
	useBias         bool    // whether to add bias term (intercept)
	deterministic   bool    // whether to use deterministic mode for reproducibility

	// Matrix data
	AInv  *mat.Dense    // precision matrix A^(-1) (d x d)
	L     *mat.TriDense // Cholesky factor such that σ²A^(-1) = L·L^T (d x d)
	b     *mat.VecDense // bias vector (d x 1)
	muVec *mat.VecDense // mean parameters μ = A^(-1) * b (d x 1)

	// Random number generation
	rng         *rand.Rand // primary RNG for reproducibility
	rngPool     *sync.Pool // pool of RNG instances for better concurrency
	seedCounter int64      // atomic counter for unique seeds

	// Multiple RNGs for deterministic mode to reduce contention
	deterministicRNGs []*rand.Rand // fixed array of RNGs for deterministic mode
	rngMutexes        []sync.Mutex // per-RNG mutexes for deterministic mode
	numRNGs           int          // number of RNGs in deterministic mode
	roundRobinCounter int64        // separate counter for round-robin RNG selection

	// Statistics and maintenance
	nUpdates        uint64 // atomic counter for number of updates
	lastMaintenance uint64 // last maintenance timestamp
	failedCholesky  uint64 // atomic counter for failed Cholesky factorizations

	// Buffer pools for performance optimization
	vectorPool sync.Pool // pool for d-dimensional vectors
	matrixPool sync.Pool // pool for d x d matrices
	floatPool  sync.Pool // pool for float64 slices
	symPool    sync.Pool // pool for d x d symmetric matrices

	// Mutex for thread safety
	mu sync.RWMutex
}

// Option defines a functional option for configuring LinTS
type Option func(*LinTS)

// WithLambda sets the regularization parameter
func WithLambda(lambda float64) Option {
	return func(l *LinTS) {
		l.lambda = lambda
	}
}

// WithSigma2 sets the noise variance for Thompson sampling
func WithSigma2(sigma2 float64) Option {
	return func(l *LinTS) {
		l.sigma2 = sigma2
	}
}

// WithMaintenanceFreq sets the frequency of numerical maintenance
func WithMaintenanceFreq(freq int) Option {
	return func(l *LinTS) {
		l.maintenanceFreq = freq
	}
}

// WithBias enables or disables the bias term
func WithBias(useBias bool) Option {
	return func(l *LinTS) {
		l.useBias = useBias
	}
}

// WithRandomSeed sets the random seed for reproducibility
func WithRandomSeed(seed int64) Option {
	return func(l *LinTS) {
		if seed == 0 {
			seed = time.Now().UnixNano()
		}
		l.rng = rand.New(rand.NewSource(seed))
		l.seedCounter = seed
	}
}

// WithDeterministic enables deterministic mode for reproducible results
func WithDeterministic(deterministic bool) Option {
	return func(l *LinTS) {
		l.deterministic = deterministic
	}
}

// NewLinTS creates a new Linear Thompson Sampling instance
func NewLinTS(dFeatures int, options ...Option) (*LinTS, error) {
	if dFeatures <= 0 {
		return nil, fmt.Errorf("feature dimension must be positive, got %d", dFeatures)
	}

	l := &LinTS{
		dFeatures:       dFeatures,
		useBias:         true,
		lambda:          1.0,
		sigma2:          1.0,
		maintenanceFreq: 5000,
		rng:             rand.New(rand.NewSource(time.Now().UnixNano())),
		seedCounter:     time.Now().UnixNano(),
	}

	// Apply options
	for _, opt := range options {
		opt(l)
	}

	// Set actual dimension (with bias if enabled)
	if l.useBias {
		l.d = l.dFeatures + 1
	} else {
		l.d = l.dFeatures
	}

	// Initialize matrices
	l.AInv = mat.NewDense(l.d, l.d, nil)
	l.L = mat.NewTriDense(l.d, mat.Lower, nil)
	l.b = mat.NewVecDense(l.d, nil)
	l.muVec = mat.NewVecDense(l.d, nil)

	// Initialize precision matrix A^(-1) = I/λ
	for i := 0; i < l.d; i++ {
		l.AInv.Set(i, i, 1.0/l.lambda)
	}

	// Initialize Cholesky factor L such that σ²A^(-1) = L·L^T
	scaleFactor := math.Sqrt(l.sigma2 / l.lambda)
	for i := 0; i < l.d; i++ {
		l.L.SetTri(i, i, scaleFactor)
	}

	// Initialize RNG pool
	l.rngPool = &sync.Pool{
		New: func() any {
			if l.deterministic {
				// In deterministic mode, use sequential seeds for reproducibility
				seed := atomic.AddInt64(&l.seedCounter, 1)
				return rand.New(rand.NewSource(seed))
			} else {
				// Non-deterministic mode with time mixing
				seed := atomic.AddInt64(&l.seedCounter, 1)
				return rand.New(rand.NewSource(seed ^ time.Now().UnixNano()))
			}
		},
	}

	// Initialize multiple RNGs for deterministic mode to reduce contention
	l.numRNGs = runtime.GOMAXPROCS(0)
	if l.deterministic {
		l.deterministicRNGs = make([]*rand.Rand, l.numRNGs)
		l.rngMutexes = make([]sync.Mutex, l.numRNGs)
		for i := 0; i < l.numRNGs; i++ {
			// Use base seed + offset for each RNG
			seed := l.seedCounter + int64(i*1000) // spread seeds apart
			l.deterministicRNGs[i] = rand.New(rand.NewSource(seed))
		}
	}

	// Initialize buffer pools
	l.vectorPool = sync.Pool{
		New: func() any {
			return mat.NewVecDense(l.d, nil)
		},
	}
	l.matrixPool = sync.Pool{
		New: func() any {
			return mat.NewDense(l.d, l.d, nil)
		},
	}
	l.floatPool = sync.Pool{
		New: func() any {
			return make([]float64, l.d)
		},
	}
	l.symPool = sync.Pool{
		New: func() any {
			return mat.NewSymDense(l.d, nil)
		},
	}

	return l, nil
}

// validateContext validates and normalizes the context vector
func (l *LinTS) validateContext(x []float64) ([]float64, error) {
	if len(x) != l.dFeatures {
		return nil, fmt.Errorf("context dimension %d != model feature dimension %d", len(x), l.dFeatures)
	}

	// Create normalized copy
	normalized := make([]float64, len(x))
	copy(normalized, x)

	// Handle invalid values
	for i, val := range normalized {
		if math.IsInf(val, 0) || math.IsNaN(val) {
			normalized[i] = 0.0
		}
	}

	// L2 normalize for better numerical stability
	norm := 0.0
	for _, val := range normalized {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	// Check for zero vectors
	if norm <= 1e-12 {
		return nil, fmt.Errorf("zero or near-zero vector detected, norm=%e", norm)
	}

	for i := range normalized {
		normalized[i] /= norm
	}

	// Add bias term if enabled (scaled by 1/√d for better conditioning)
	if l.useBias {
		biasScale := 1.0 / math.Sqrt(math.Max(1, float64(l.dFeatures)))
		normalized = append(normalized, biasScale)
	}

	return normalized, nil
}

// contextToVec converts context slice to gonum vector
func (l *LinTS) contextToVec(ctx []float64) *mat.VecDense {
	// Get vector from pool
	vec := l.vectorPool.Get().(*mat.VecDense)

	// Reset vector to ensure clean state before reuse
	vec.Reset()
	vec.ReuseAsVec(l.d)

	// Efficient copy: only set what we need
	data := vec.RawVector().Data
	n := len(ctx)
	if n > l.d {
		n = l.d
	}

	// Copy context data efficiently
	copy(data[:n], ctx[:n])

	// Zero remaining elements (if any) only if necessary
	if n < l.d {
		for i := n; i < l.d; i++ {
			data[i] = 0.0
		}
	}

	return vec
}

// returnVector returns a vector to the pool
func (l *LinTS) returnVector(v *mat.VecDense) {
	l.vectorPool.Put(v)
}

// returnMatrix returns a matrix to the pool
func (l *LinTS) returnMatrix(m *mat.Dense) {
	l.matrixPool.Put(m)
}

// returnSymMatrix returns a symmetric matrix to the pool
func (l *LinTS) returnSymMatrix(m *mat.SymDense) {
	l.symPool.Put(m)
}

// sampleTheta samples θ ~ N(μ, σ²A^(-1)) using precomputed Cholesky factor
func (l *LinTS) sampleTheta() *mat.VecDense {
	l.mu.RLock()
	defer l.mu.RUnlock()

	// Use buffer to avoid Reset() allocations
	zBuf := l.floatPool.Get().([]float64)
	defer func() {
		// Only return to pool if size is reasonable to prevent contamination
		if cap(zBuf) <= l.d*4 {
			l.floatPool.Put(zBuf)
		}
	}()
	if cap(zBuf) < l.d || cap(zBuf) > l.d*4 {
		zBuf = make([]float64, l.d)
	} else {
		zBuf = zBuf[:l.d]
	}

	// Generate standard normal samples
	if l.deterministic && len(l.deterministicRNGs) > 0 {
		// In deterministic mode, use one of multiple RNGs to reduce contention
		// Use separate counter for round-robin to avoid seed collisions
		rngIdx := int(atomic.AddInt64(&l.roundRobinCounter, 1)) % l.numRNGs
		rng := l.deterministicRNGs[rngIdx]

		// Use per-RNG mutex to reduce contention
		l.rngMutexes[rngIdx].Lock()
		for i := 0; i < l.d; i++ {
			zBuf[i] = rng.NormFloat64()
		}
		l.rngMutexes[rngIdx].Unlock()
	} else {
		// In non-deterministic mode, use pooled RNG for better concurrency
		rng := l.rngPool.Get().(*rand.Rand)
		for i := 0; i < l.d; i++ {
			zBuf[i] = rng.NormFloat64()
		}
		l.rngPool.Put(rng)
	}

	// Get vectors from pool
	z := l.vectorPool.Get().(*mat.VecDense)
	z.Reset()
	z.ReuseAsVec(l.d)
	copy(z.RawVector().Data, zBuf)

	// θ = μ + L @ z
	theta := l.vectorPool.Get().(*mat.VecDense)
	theta.Reset()
	theta.ReuseAsVec(l.d)
	theta.MulVec(l.L, z)
	theta.AddVec(theta, l.muVec)

	// Return z to pool, but keep theta for caller
	l.returnVector(z)
	return theta
}

// Score computes Thompson sampling score for a single context
func (l *LinTS) Score(x []float64) (float64, error) {
	ctx, err := l.validateContext(x)
	if err != nil {
		return 0, err
	}

	ctxVec := l.contextToVec(ctx)
	defer l.returnVector(ctxVec)

	theta := l.sampleTheta()
	defer l.returnVector(theta)

	score := mat.Dot(ctxVec, theta)
	return score, nil
}

// ScoreBatch computes Thompson sampling scores for multiple contexts using shared θ sample
func (l *LinTS) ScoreBatch(X [][]float64) ([]float64, error) {
	if len(X) == 0 {
		return nil, fmt.Errorf("empty context batch")
	}

	// Validate and normalize all contexts
	normalizedX := make([][]float64, len(X))
	for i, x := range X {
		var err error
		normalizedX[i], err = l.validateContext(x)
		if err != nil {
			return nil, fmt.Errorf("context %d: %w", i, err)
		}
	}

	// Sample θ once for the entire batch
	theta := l.sampleTheta()
	defer l.returnVector(theta)

	// Compute scores
	scores := make([]float64, len(X))
	ctxVecs := make([]*mat.VecDense, len(X))

	// Create all context vectors first
	for i, ctx := range normalizedX {
		ctxVecs[i] = l.contextToVec(ctx)
	}

	// Compute all scores
	for i, ctxVec := range ctxVecs {
		scores[i] = mat.Dot(ctxVec, theta)
	}

	// Return all context vectors to pool
	for _, ctxVec := range ctxVecs {
		l.returnVector(ctxVec)
	}

	return scores, nil
}

// cholRank1Update performs rank-1 Cholesky update: L ← cholupdate(L, ±v)
func (l *LinTS) cholRank1Update(v *mat.VecDense, subtract bool) error {
	sign := 1.0
	if subtract {
		sign = -1.0
	}

	vData := make([]float64, l.d)
	for i := 0; i < l.d; i++ {
		vData[i] = v.AtVec(i)
	}

	for k := 0; k < l.d; k++ {
		// Check SPD preservation for downdate
		if subtract && l.L.At(k, k)*l.L.At(k, k) <= vData[k]*vData[k] {
			return fmt.Errorf("downdate would destroy SPD property at k=%d", k)
		}

		// Givens rotation to eliminate v[k]
		r := math.Sqrt(l.L.At(k, k)*l.L.At(k, k) + sign*vData[k]*vData[k])

		if math.Abs(r) < 1e-15 {
			return fmt.Errorf("numerical instability in Cholesky update at k=%d, r=%e", k, r)
		}

		lkk := l.L.At(k, k)
		if math.Abs(lkk) < 1e-15 {
			return fmt.Errorf("zero diagonal element in Cholesky factor at k=%d", k)
		}

		c := r / lkk
		s := vData[k] / lkk

		// Update diagonal element
		l.L.SetTri(k, k, r)

		if k+1 < l.d {
			// Update remaining elements in column k
			for i := k + 1; i < l.d; i++ {
				newVal := (l.L.At(i, k) + sign*s*vData[i]) / c
				l.L.SetTri(i, k, newVal)
			}

			// Update v for next iteration
			for i := k + 1; i < l.d; i++ {
				vData[i] = c*vData[i] - s*l.L.At(i, k)
			}
		}
	}

	return nil
}

// recomputeCholesky recomputes Cholesky factor from scratch (fallback)
func (l *LinTS) recomputeCholesky() error {
	// Get covariance matrix from pool for efficiency
	cov := l.matrixPool.Get().(*mat.Dense)
	defer l.returnMatrix(cov)
	cov.Reset()
	cov.ReuseAs(l.d, l.d)
	cov.Scale(l.sigma2, l.AInv)

	// Add dosed regularization for numerical stability (only when needed)
	for i := 0; i < l.d; i++ {
		if cov.At(i, i) < 1e-12 {
			cov.Set(i, i, 1e-12)
		}
	}

	// Try full Cholesky decomposition using SymDense matrix (correct approach)
	// Validate dimension before creating Cholesky
	if l.d <= 0 {
		return fmt.Errorf("invalid dimension for Cholesky: d=%d", l.d)
	}

	// Get SymDense matrix from pool and copy upper triangle from Dense matrix
	sym := l.symPool.Get().(*mat.SymDense)
	defer l.returnSymMatrix(sym)

	// Reset the symmetric matrix for clean state
	sym.Reset()
	sym.ReuseAsSym(l.d)

	for i := 0; i < l.d; i++ {
		for j := i; j < l.d; j++ {
			// Symmetrize the value and set it
			val := 0.5 * (cov.At(i, j) + cov.At(j, i))
			sym.SetSym(i, j, val)
		}
	}

	var chol mat.Cholesky
	if chol.Factorize(sym) {
		chol.LTo(l.L) // direct transfer to lower triangular
		return nil
	}

	// Diagonal fallback - increment counter for monitoring
	atomic.AddUint64(&l.failedCholesky, 1)
	for i := 0; i < l.d; i++ {
		// Extract diagonal element from covariance matrix
		diag := math.Sqrt(math.Max(cov.At(i, i), 1e-12))
		for j := 0; j <= i; j++ {
			if i == j {
				l.L.SetTri(i, j, diag)
			} else {
				l.L.SetTri(i, j, 0.0)
			}
		}
	}
	return nil
}

// Update updates the model with a single context-reward pair
func (l *LinTS) Update(x []float64, reward float64) error {
	ctx, err := l.validateContext(x)
	if err != nil {
		return err
	}

	// Validate and clip reward
	if math.IsInf(reward, 0) || math.IsNaN(reward) {
		reward = 0.0
	}
	reward = math.Max(-10.0, math.Min(10.0, reward)) // Clip to [-10, 10]

	l.mu.Lock()
	defer l.mu.Unlock()

	ctxVec := l.contextToVec(ctx)
	defer l.returnVector(ctxVec)

	// Sherman-Morrison update: A^(-1) ← A^(-1) - (A^(-1)xx^T A^(-1))/(1 + x^T A^(-1) x)
	Ax := l.vectorPool.Get().(*mat.VecDense)
	defer l.returnVector(Ax)
	Ax.Reset()
	Ax.ReuseAsVec(l.d)
	Ax.MulVec(l.AInv, ctxVec)

	denominator := mat.Dot(ctxVec, Ax) + 1.0
	denominator = math.Max(denominator, 1e-12) // Safe denominator

	// Rank-1 update of A^(-1)
	outerProd := l.matrixPool.Get().(*mat.Dense)
	defer l.returnMatrix(outerProd)
	outerProd.Reset()
	outerProd.ReuseAs(l.d, l.d)
	outerProd.Outer(1.0/denominator, Ax, Ax)
	l.AInv.Sub(l.AInv, outerProd)

	// Update bias vector with real reward
	scaledCtx := l.vectorPool.Get().(*mat.VecDense)
	defer l.returnVector(scaledCtx)
	scaledCtx.Reset()
	scaledCtx.ReuseAsVec(l.d)
	scaledCtx.ScaleVec(reward, ctxVec)
	l.b.AddVec(l.b, scaledCtx)

	// Update mean parameters μ = A^(-1) * b
	l.muVec.MulVec(l.AInv, l.b)

	// O(d²) rank-1 Cholesky update
	err = l.updateCholesky(Ax, denominator)
	if err != nil {
		// Fallback to full recomputation
		l.recomputeCholesky()
	}

	// Update statistics
	atomic.AddUint64(&l.nUpdates, 1)

	// Numerical maintenance
	nUpdates := atomic.LoadUint64(&l.nUpdates)
	if nUpdates-atomic.LoadUint64(&l.lastMaintenance) >= uint64(l.maintenanceFreq) {
		l.numericalMaintenance()
	}

	return nil
}

// updateCholesky performs the Cholesky update for covariance matrix update
func (l *LinTS) updateCholesky(Ax *mat.VecDense, denominator float64) error {
	// For covariance update: Σ_new = Σ - (Σx)(Σx)^T/(1 + x^TΣx)
	// where Σ = σ²A^(-1), so Σx = σ²Ax
	// Downdate vector: u = Σx / √(1 + x^TΣx) = σ²Ax / √(1 + x^TΣx)
	sigma2Ax := l.vectorPool.Get().(*mat.VecDense)
	defer l.returnVector(sigma2Ax)
	sigma2Ax.Reset()
	sigma2Ax.ReuseAsVec(l.d)
	sigma2Ax.ScaleVec(l.sigma2, Ax)

	denSigma := 1.0 + l.sigma2*(denominator-1.0) // 1 + x^TΣx = 1 + σ²(x^TA^(-1)x)
	updateVector := l.vectorPool.Get().(*mat.VecDense)
	defer l.returnVector(updateVector)
	updateVector.Reset()
	updateVector.ReuseAsVec(l.d)
	updateVector.ScaleVec(1.0/math.Sqrt(denSigma), sigma2Ax)

	return l.cholRank1Update(updateVector, true)
}

// numericalMaintenance performs periodic numerical maintenance
func (l *LinTS) numericalMaintenance() {
	// Symmetrize A^(-1) to counter floating-point drift
	for i := 0; i < l.d; i++ {
		for j := i + 1; j < l.d; j++ {
			avg := 0.5 * (l.AInv.At(i, j) + l.AInv.At(j, i))
			l.AInv.Set(i, j, avg)
			l.AInv.Set(j, i, avg)
		}
	}

	// Check condition number and regularize if needed
	var svd mat.SVD
	if svd.Factorize(l.AInv, mat.SVDThin) {
		values := svd.Values(nil)
		if len(values) > 0 {
			condNum := values[0] / values[len(values)-1]
			if condNum > 1e10 {
				// Add regularization
				for i := 0; i < l.d; i++ {
					l.AInv.Set(i, i, l.AInv.At(i, i)+1e-8)
				}
			}
		}
	}

	// Check for numerical issues in mean
	for i := 0; i < l.d; i++ {
		if math.IsInf(l.muVec.AtVec(i), 0) || math.IsNaN(l.muVec.AtVec(i)) {
			l.muVec.SetVec(i, 0.0)
		}
		if math.IsInf(l.b.AtVec(i), 0) || math.IsNaN(l.b.AtVec(i)) {
			l.b.SetVec(i, 0.0)
		}
	}

	// Recompute Cholesky factor
	l.recomputeCholesky()

	atomic.StoreUint64(&l.lastMaintenance, atomic.LoadUint64(&l.nUpdates))
}

// GetStats returns current model statistics
func (l *LinTS) GetStats() map[string]any {
	l.mu.RLock()
	defer l.mu.RUnlock()

	var svd mat.SVD
	condNum := math.NaN()
	if svd.Factorize(l.AInv, mat.SVDThin) {
		values := svd.Values(nil)
		if len(values) > 0 {
			condNum = values[0] / values[len(values)-1]
		}
	}

	muNorm := mat.Norm(l.muVec, 2)
	traceAInv := mat.Trace(l.AInv)

	// Calculate Cholesky norm manually to avoid zero dimension issues
	cholNorm := 0.0
	for i := 0; i < l.d; i++ {
		for j := 0; j <= i; j++ {
			val := l.L.At(i, j)
			cholNorm += val * val
		}
	}
	cholNorm = math.Sqrt(cholNorm)

	return map[string]any{
		"n_updates":        atomic.LoadUint64(&l.nUpdates),
		"condition_number": condNum,
		"norm_mu":          muNorm,
		"trace_A_inv":      traceAInv,
		"last_maintenance": atomic.LoadUint64(&l.lastMaintenance),
		"failed_cholesky":  atomic.LoadUint64(&l.failedCholesky),
		"cholesky_norm":    cholNorm,
		"d_features":       l.dFeatures,
		"d":                l.d,
		"use_bias":         l.useBias,
		"deterministic":    l.deterministic,
		"num_rngs":         l.numRNGs,
		"lambda":           l.lambda,
		"sigma2":           l.sigma2,
	}
}

// Reset resets the model to its initial state (useful for A/B testing)
func (l *LinTS) Reset() {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Reset precision matrix A^(-1) = I/λ
	for i := 0; i < l.d; i++ {
		for j := 0; j < l.d; j++ {
			if i == j {
				l.AInv.Set(i, j, 1.0/l.lambda)
			} else {
				l.AInv.Set(i, j, 0.0)
			}
		}
	}

	// Reset Cholesky factor L such that σ²A^(-1) = L·L^T
	scaleFactor := math.Sqrt(l.sigma2 / l.lambda)
	for i := 0; i < l.d; i++ {
		for j := 0; j <= i; j++ {
			if i == j {
				l.L.SetTri(i, j, scaleFactor)
			} else {
				l.L.SetTri(i, j, 0.0)
			}
		}
	}

	// Reset bias vector and mean
	for i := 0; i < l.d; i++ {
		l.b.SetVec(i, 0.0)
		l.muVec.SetVec(i, 0.0)
	}

	// Reset counters
	atomic.StoreUint64(&l.nUpdates, 0)
	atomic.StoreUint64(&l.lastMaintenance, 0)
	atomic.StoreUint64(&l.failedCholesky, 0)
}

// LinTSState represents the serializable state of LinTS
type LinTSState struct {
	Version         int       `gob:"version"`
	DFeatures       int       `gob:"d_features"`
	D               int       `gob:"d"`
	Lambda          float64   `gob:"lambda"`
	Sigma2          float64   `gob:"sigma2"`
	MaintenanceFreq int       `gob:"maintenance_freq"`
	UseBias         bool      `gob:"use_bias"`
	Deterministic   bool      `gob:"deterministic"`
	AInvData        []float64 `gob:"ainv_data"`
	BData           []float64 `gob:"b_data"`
	MuVecData       []float64 `gob:"mu_vec_data"`
	NUpdates        uint64    `gob:"n_updates"`
	LastMaintenance uint64    `gob:"last_maintenance"`
	FailedCholesky  uint64    `gob:"failed_cholesky"`
}

// Save serializes the model state to gob format
func (l *LinTS) Save(w io.Writer) error {
	l.mu.RLock()
	defer l.mu.RUnlock()

	state := LinTSState{
		Version:         1,
		DFeatures:       l.dFeatures,
		D:               l.d,
		Lambda:          l.lambda,
		Sigma2:          l.sigma2,
		MaintenanceFreq: l.maintenanceFreq,
		UseBias:         l.useBias,
		Deterministic:   l.deterministic,
		NUpdates:        atomic.LoadUint64(&l.nUpdates),
		LastMaintenance: atomic.LoadUint64(&l.lastMaintenance),
		FailedCholesky:  atomic.LoadUint64(&l.failedCholesky),
	}

	// Copy AInv data (flattened)
	aInvRaw := l.AInv.RawMatrix()
	state.AInvData = make([]float64, len(aInvRaw.Data))
	copy(state.AInvData, aInvRaw.Data)

	// Copy b data
	bRaw := l.b.RawVector()
	state.BData = make([]float64, len(bRaw.Data))
	copy(state.BData, bRaw.Data)

	// Copy muVec data
	muVecRaw := l.muVec.RawVector()
	state.MuVecData = make([]float64, len(muVecRaw.Data))
	copy(state.MuVecData, muVecRaw.Data)

	encoder := gob.NewEncoder(w)
	return encoder.Encode(state)
}

// Load deserializes model state from gob format
func Load(r io.Reader, seed int64) (*LinTS, error) {
	decoder := gob.NewDecoder(r)

	var state LinTSState
	if err := decoder.Decode(&state); err != nil {
		return nil, err
	}

	if state.Version != 1 {
		return nil, errors.New("unsupported gob version")
	}

	// Create new instance with the same configuration
	options := []Option{
		WithLambda(state.Lambda),
		WithSigma2(state.Sigma2),
		WithMaintenanceFreq(state.MaintenanceFreq),
		WithBias(state.UseBias),
		WithDeterministic(state.Deterministic),
		WithRandomSeed(seed),
	}

	l, err := NewLinTS(state.DFeatures, options...)
	if err != nil {
		return nil, err
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	// Validate data lengths
	expectedAInvLen := state.D * state.D
	if len(state.AInvData) != expectedAInvLen {
		return nil, errors.New("invalid AInv data length")
	}
	if len(state.BData) != state.D {
		return nil, errors.New("invalid b data length")
	}
	if len(state.MuVecData) != state.D {
		return nil, errors.New("invalid muVec data length")
	}

	// Restore AInv
	aInvData := make([]float64, expectedAInvLen)
	copy(aInvData, state.AInvData)
	l.AInv = mat.NewDense(state.D, state.D, aInvData)

	// Restore b
	bData := make([]float64, state.D)
	copy(bData, state.BData)
	l.b = mat.NewVecDense(state.D, bData)

	// Restore muVec
	muVecData := make([]float64, state.D)
	copy(muVecData, state.MuVecData)
	l.muVec = mat.NewVecDense(state.D, muVecData)

	// Restore counters
	atomic.StoreUint64(&l.nUpdates, state.NUpdates)
	atomic.StoreUint64(&l.lastMaintenance, state.LastMaintenance)
	atomic.StoreUint64(&l.failedCholesky, state.FailedCholesky)

	// Recompute Cholesky factor from the precision matrix
	if err := l.recomputeCholesky(); err != nil {
		return nil, err
	}

	return l, nil
}
