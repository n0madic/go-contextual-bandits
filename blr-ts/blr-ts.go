package blrts

import (
	"encoding/gob"
	"errors"
	"io"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"gonum.org/v1/gonum/mat"
)

// BLRTS implements Bayesian Linear Regression Thompson Sampling for contextual bandits.
type BLRTS struct {
	nArms       int
	contextDim  int
	alpha       float64
	sigma       float64
	rng         *rand.Rand // For reproducibility (legacy)
	rngPool     *sync.Pool // Pool of RNG instances for better concurrency
	seedCounter int64      // Atomic counter for unique seeds

	mu        []*mat.VecDense // Posterior means
	cov       []*mat.Dense    // Posterior covariances
	chol      []*mat.Dense    // Cholesky factors
	outerBuf  []*mat.Dense    // Buffers for outer products
	vBuf      []*mat.VecDense // Buffers for vectors
	scaleBuf  []*mat.Dense    // Buffers for scaling operations
	xBuf      []*mat.VecDense // Buffers for context vectors in Update
	AxBuf     []*mat.VecDense // Buffers for A*x vectors in Update
	updateBuf []*mat.VecDense // Buffers for update vectors in Update

	// SelectAction buffers to avoid allocations
	zBuf    [][]float64 // Pre-allocated random normal buffers
	LtCBuf  [][]float64 // Pre-allocated L^T*c buffers
	rewards []float64   // Pre-allocated rewards buffer

	armLocks []sync.RWMutex // Per-arm mutexes for better parallelism
}

// NewBLRTS creates a new instance of BLRTS.
func NewBLRTS(nArms, contextDim int, alpha, sigma float64, seed int64) (*BLRTS, error) {
	var rng *rand.Rand
	if seed == 0 {
		rng = rand.New(rand.NewSource(rand.Int63()))
	} else {
		rng = rand.New(rand.NewSource(seed))
	}

	b := &BLRTS{
		nArms:       nArms,
		contextDim:  contextDim,
		alpha:       alpha,
		sigma:       sigma,
		rng:         rng,
		seedCounter: time.Now().UnixNano(), // Initialize with current time
		mu:          make([]*mat.VecDense, nArms),
		cov:         make([]*mat.Dense, nArms),
		chol:        make([]*mat.Dense, nArms),
		outerBuf:    make([]*mat.Dense, nArms),
		vBuf:        make([]*mat.VecDense, nArms),
		scaleBuf:    make([]*mat.Dense, nArms),
		xBuf:        make([]*mat.VecDense, nArms),
		AxBuf:       make([]*mat.VecDense, nArms),
		updateBuf:   make([]*mat.VecDense, nArms),
		zBuf:        make([][]float64, nArms),
		LtCBuf:      make([][]float64, nArms),
		rewards:     make([]float64, nArms),
		armLocks:    make([]sync.RWMutex, nArms),
	}

	// Initialize RNG pool after b is created
	b.rngPool = &sync.Pool{
		New: func() interface{} {
			// Use atomic counter to avoid seed collisions
			seed := atomic.AddInt64(&b.seedCounter, 1)
			return rand.New(rand.NewSource(seed ^ time.Now().UnixNano()))
		},
	}

	for a := 0; a < nArms; a++ {
		b.mu[a] = mat.NewVecDense(contextDim, nil) // Zeros
		b.cov[a] = mat.NewDense(contextDim, contextDim, nil)
		for i := 0; i < contextDim; i++ {
			b.cov[a].Set(i, i, alpha)
		}
		if chol, err := b.safeChol(b.cov[a]); err != nil {
			return nil, err
		} else {
			b.chol[a] = chol
		}
		b.outerBuf[a] = mat.NewDense(contextDim, contextDim, nil)
		b.vBuf[a] = mat.NewVecDense(contextDim, nil)
		b.scaleBuf[a] = mat.NewDense(contextDim, contextDim, nil)
		b.xBuf[a] = mat.NewVecDense(contextDim, nil)
		b.AxBuf[a] = mat.NewVecDense(contextDim, nil)
		b.updateBuf[a] = mat.NewVecDense(contextDim, nil)
		b.zBuf[a] = make([]float64, contextDim)
		b.LtCBuf[a] = make([]float64, contextDim)
	}

	return b, nil
}

func (b *BLRTS) denseToSym(d *mat.Dense) *mat.SymDense {
	n, _ := d.Dims()
	sym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sym.SetSym(i, j, d.At(i, j))
		}
	}
	return sym
}

func (b *BLRTS) safeChol(C *mat.Dense) (*mat.Dense, error) {
	sym := b.denseToSym(C)
	var chol mat.Cholesky
	if ok := chol.Factorize(sym); ok {
		Ltri := mat.NewTriDense(b.contextDim, mat.Lower, nil)
		chol.LTo(Ltri)
		L := mat.NewDense(b.contextDim, b.contextDim, nil)
		L.Copy(Ltri)
		return L, nil
	}

	// Adaptive jitter
	jittered := mat.NewDense(b.contextDim, b.contextDim, nil)
	jittered.Copy(C)
	trace := 0.0
	for i := 0; i < b.contextDim; i++ {
		trace += jittered.At(i, i)
	}
	eps := 1e-8 * trace / float64(b.contextDim)
	for i := 0; i < b.contextDim; i++ {
		jittered.Set(i, i, jittered.At(i, i)+eps)
	}

	sym = b.denseToSym(jittered)
	if ok := chol.Factorize(sym); ok {
		Ltri := mat.NewTriDense(b.contextDim, mat.Lower, nil)
		chol.LTo(Ltri)
		L := mat.NewDense(b.contextDim, b.contextDim, nil)
		L.Copy(Ltri)
		return L, nil
	}
	return nil, errors.New("cholesky factorization failed even with jitter")
}

func (b *BLRTS) cholRank1Downdate(L *mat.Dense, v *mat.VecDense) (*mat.Dense, error) {
	n := b.contextDim
	Lcopy := mat.NewDense(n, n, nil)
	Lcopy.Copy(L)

	vCopy := mat.NewVecDense(n, nil)
	vCopy.CopyVec(v)

	for k := 0; k < n; k++ {
		r := math.Hypot(Lcopy.At(k, k), vCopy.AtVec(k))
		if r < 1e-12 || math.IsNaN(r) {
			return nil, errors.New("failed to downdate Cholesky factorization; falling back to full recomputation")
		}
		c := r / Lcopy.At(k, k)
		s := vCopy.AtVec(k) / Lcopy.At(k, k)
		Lcopy.Set(k, k, r)
		if k+1 < n {
			// Update subcolumn
			for i := k + 1; i < n; i++ {
				Lcopy.Set(i, k, (Lcopy.At(i, k)+s*vCopy.AtVec(i))/c)
			}
			// Update v subvector
			for i := k + 1; i < n; i++ {
				vCopy.SetVec(i, c*vCopy.AtVec(i)-s*Lcopy.At(i, k))
			}
		}
	}
	// Ensure strictly lower triangular in-place (zero upper part)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			Lcopy.Set(i, j, 0.0)
		}
	}
	return Lcopy, nil
}

// SelectAction selects an arm using Thompson Sampling.
func (b *BLRTS) SelectAction(context mat.Vector) (int, error) {
	if context.Len() != b.contextDim {
		return 0, errors.New("context dimension mismatch")
	}

	// Create local context buffer to avoid race conditions
	contextBuf := make([]float64, b.contextDim)
	for i := 0; i < b.contextDim; i++ {
		contextBuf[i] = context.AtVec(i)
	}

	// Sample and score each arm using local buffers (avoids shared data races)
	rng := b.rngPool.Get().(*rand.Rand)
	defer b.rngPool.Put(rng)

	// Local buffers for sampling and projection (one allocation each)
	zBuf := make([]float64, b.contextDim)
	ltCBuf := make([]float64, b.contextDim)

	// Compute rewards for each arm
	maxIdx := 0
	maxVal := math.Inf(-1)
	for a := 0; a < b.nArms; a++ {
		// sample independent normals for this arm
		for i := 0; i < b.contextDim; i++ {
			zBuf[i] = rng.NormFloat64()
		}
		b.armLocks[a].RLock()

		// Compute mu^T * context
		muDotContext := 0.0
		muRaw := b.mu[a].RawVector().Data
		for i := 0; i < b.contextDim; i++ {
			muDotContext += muRaw[i] * contextBuf[i]
		}

		// Compute L^T * context into local buffer
		cholRaw := b.chol[a].RawMatrix()
		for i := 0; i < b.contextDim; i++ {
			ltCBuf[i] = 0.0
			for j := 0; j < b.contextDim; j++ {
				// L^T[i][j] = L[j][i]
				ltCBuf[i] += cholRaw.Data[j*cholRaw.Stride+i] * contextBuf[j]
			}
		}

		// Compute z^T * (L^T * context)
		zDotLtC := 0.0
		for i := 0; i < b.contextDim; i++ {
			zDotLtC += zBuf[i] * ltCBuf[i]
		}

		reward := muDotContext + zDotLtC
		b.armLocks[a].RUnlock()

		if reward > maxVal {
			maxVal = reward
			maxIdx = a
		}
	}

	return maxIdx, nil
}

// Update updates the posterior for the selected arm.
func (b *BLRTS) Update(arm int, context mat.Vector, reward float64) error {
	if context.Len() != b.contextDim {
		return errors.New("context dimension mismatch")
	}

	b.armLocks[arm].Lock()
	defer b.armLocks[arm].Unlock()

	beta := 1.0 / (b.sigma * b.sigma)

	// Use pre-allocated buffers instead of creating new VecDense
	x := b.xBuf[arm]
	x.CopyVec(context)

	Ax := b.AxBuf[arm]
	Ax.MulVec(b.cov[arm], x)

	denom := 1.0 + beta*mat.Dot(x, Ax)
	if denom < 1e-12 {
		return nil
	}
	denom = math.Max(denom, 1e-12)
	coef := beta / denom
	if coef < 1e-12 {
		return nil
	}

	// Outer product using buffer
	outerBuf := b.outerBuf[arm]
	outerBuf.Outer(1.0, Ax, Ax)

	// Update cov: cov -= coef * outerBuf (optimized in-place scaling)
	outerBuf.Scale(coef, outerBuf)
	b.cov[arm].Sub(b.cov[arm], outerBuf)

	// Symmetrize in-place
	scaleBuf := b.scaleBuf[arm]
	scaleBuf.Copy(b.cov[arm].T())
	b.cov[arm].Add(b.cov[arm], scaleBuf)
	b.cov[arm].Scale(0.5, b.cov[arm])

	// Rank-1 downdate - use existing vBuf, no need for vCopy
	v := b.vBuf[arm]
	v.ScaleVec(math.Sqrt(coef), Ax)

	res, err := b.cholRank1Downdate(b.chol[arm], v)
	if err != nil {
		if chol, cholErr := b.safeChol(b.cov[arm]); cholErr != nil {
			return cholErr
		} else {
			b.chol[arm] = chol
		}
	} else {
		b.chol[arm] = res
	}

	// Update mean using pre-allocated buffer
	diff := reward - mat.Dot(x, b.mu[arm])
	updateVec := b.updateBuf[arm]
	updateVec.ScaleVec(coef*diff, Ax)
	b.mu[arm].AddVec(b.mu[arm], updateVec)

	// Reset buffers to prevent float drift accumulation
	updateVec.Zero()
	b.vBuf[arm].Zero()

	return nil
}

// ResetArm resets a specific arm to its prior state (useful for A/B experiments)
func (b *BLRTS) ResetArm(arm int) error {
	if arm < 0 || arm >= b.nArms {
		return errors.New("invalid arm index")
	}

	b.armLocks[arm].Lock()
	defer b.armLocks[arm].Unlock()

	// Reset to prior: μ = 0, Σ = αI
	b.mu[arm].Zero()
	b.cov[arm].Zero()
	for i := 0; i < b.contextDim; i++ {
		b.cov[arm].Set(i, i, b.alpha)
	}

	// Recompute Cholesky factorization
	if chol, err := b.safeChol(b.cov[arm]); err != nil {
		return err
	} else {
		b.chol[arm] = chol
	}

	// Clear all buffers for this arm
	b.outerBuf[arm].Zero()
	b.vBuf[arm].Zero()
	b.scaleBuf[arm].Zero()
	b.xBuf[arm].Zero()
	b.AxBuf[arm].Zero()
	b.updateBuf[arm].Zero()

	// Clear float buffers
	for i := range b.zBuf[arm] {
		b.zBuf[arm][i] = 0.0
	}
	for i := range b.LtCBuf[arm] {
		b.LtCBuf[arm][i] = 0.0
	}

	return nil
}

// BLRState represents the serializable state of BLRTS
type BLRState struct {
	Version    int         `gob:"version"`
	NArms      int         `gob:"n_arms"`
	ContextDim int         `gob:"context_dim"`
	Alpha      float64     `gob:"alpha"`
	Sigma      float64     `gob:"sigma"`
	MuData     [][]float64 `gob:"mu_data"`  // Raw posterior means
	CovData    [][]float64 `gob:"cov_data"` // Raw covariance matrices (flattened)
}

// Save serializes the model state to gob format
// Note: Cholesky factors are not serialized as they can be recomputed from covariance
func (b *BLRTS) Save(w io.Writer) error {
	state := BLRState{
		Version:    1, // Gob version
		NArms:      b.nArms,
		ContextDim: b.contextDim,
		Alpha:      b.alpha,
		Sigma:      b.sigma,
		MuData:     make([][]float64, b.nArms),
		CovData:    make([][]float64, b.nArms),
	}

	// Extract raw data from matrices
	for a := 0; a < b.nArms; a++ {
		b.armLocks[a].RLock()

		// Copy mu data
		muRaw := b.mu[a].RawVector()
		state.MuData[a] = make([]float64, len(muRaw.Data))
		copy(state.MuData[a], muRaw.Data)

		// Copy cov data (flattened)
		covRaw := b.cov[a].RawMatrix()
		state.CovData[a] = make([]float64, len(covRaw.Data))
		copy(state.CovData[a], covRaw.Data)

		b.armLocks[a].RUnlock()
	}

	encoder := gob.NewEncoder(w)
	return encoder.Encode(state)
}

// Load deserializes model state from gob format
func Load(r io.Reader, seed int64) (*BLRTS, error) {
	decoder := gob.NewDecoder(r)

	var state BLRState
	if err := decoder.Decode(&state); err != nil {
		return nil, err
	}

	if state.Version != 1 {
		return nil, errors.New("unsupported gob version")
	}

	// Create new instance
	b, err := NewBLRTS(state.NArms, state.ContextDim, state.Alpha, state.Sigma, seed)
	if err != nil {
		return nil, err
	}

	// Restore state
	for a := 0; a < state.NArms; a++ {
		b.armLocks[a].Lock()

		// Restore mu
		if len(state.MuData[a]) != state.ContextDim {
			b.armLocks[a].Unlock()
			return nil, errors.New("invalid mu data length")
		}
		muData := make([]float64, state.ContextDim)
		copy(muData, state.MuData[a])
		b.mu[a] = mat.NewVecDense(state.ContextDim, muData)

		// Restore cov
		expectedCovLen := state.ContextDim * state.ContextDim
		if len(state.CovData[a]) != expectedCovLen {
			b.armLocks[a].Unlock()
			return nil, errors.New("invalid cov data length")
		}
		covData := make([]float64, expectedCovLen)
		copy(covData, state.CovData[a])
		b.cov[a] = mat.NewDense(state.ContextDim, state.ContextDim, covData)

		// Recompute Cholesky factorization
		if chol, err := b.safeChol(b.cov[a]); err != nil {
			b.armLocks[a].Unlock()
			return nil, err
		} else {
			b.chol[a] = chol
		}

		b.armLocks[a].Unlock()
	}

	return b, nil
}
