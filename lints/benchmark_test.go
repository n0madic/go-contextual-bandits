package lints

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
)

// BenchmarkLinTSPerformance tests performance across different scenarios
func BenchmarkLinTSPerformance(b *testing.B) {
	dimensions := []int{5, 10, 50, 100, 200}

	for _, d := range dimensions {
		b.Run(fmt.Sprintf("Score_d%d", d), func(b *testing.B) {
			benchmarkScore(b, d)
		})

		b.Run(fmt.Sprintf("Update_d%d", d), func(b *testing.B) {
			benchmarkUpdate(b, d)
		})

		b.Run(fmt.Sprintf("ScoreBatch_d%d", d), func(b *testing.B) {
			benchmarkScoreBatch(b, d, 100)
		})
	}
}

func benchmarkScore(b *testing.B, d int) {
	lts, err := NewLinTS(d, WithRandomSeed(42))
	if err != nil {
		b.Fatalf("NewLinTS() error = %v", err)
	}

	// Generate random context
	rng := rand.New(rand.NewSource(42))
	context := make([]float64, d)
	for i := range context {
		context[i] = rng.NormFloat64()
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := lts.Score(context)
		if err != nil {
			b.Fatalf("Score() error = %v", err)
		}
	}
}

func benchmarkUpdate(b *testing.B, d int) {
	lts, err := NewLinTS(d, WithRandomSeed(42))
	if err != nil {
		b.Fatalf("NewLinTS() error = %v", err)
	}

	// Generate random context and reward
	rng := rand.New(rand.NewSource(42))
	contexts := make([][]float64, b.N)
	rewards := make([]float64, b.N)

	for i := 0; i < b.N; i++ {
		contexts[i] = make([]float64, d)
		for j := range contexts[i] {
			contexts[i][j] = rng.NormFloat64()
		}
		rewards[i] = rng.NormFloat64()
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := lts.Update(contexts[i], rewards[i])
		if err != nil {
			b.Fatalf("Update() error = %v", err)
		}
	}
}

func benchmarkScoreBatch(b *testing.B, d int, batchSize int) {
	lts, err := NewLinTS(d, WithRandomSeed(42))
	if err != nil {
		b.Fatalf("NewLinTS() error = %v", err)
	}

	// Generate random contexts
	rng := rand.New(rand.NewSource(42))
	contexts := make([][]float64, batchSize)
	for i := range contexts {
		contexts[i] = make([]float64, d)
		for j := range contexts[i] {
			contexts[i][j] = rng.NormFloat64()
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := lts.ScoreBatch(contexts)
		if err != nil {
			b.Fatalf("ScoreBatch() error = %v", err)
		}
	}
}

// BenchmarkConcurrency tests performance under concurrent load
func BenchmarkConcurrency(b *testing.B) {
	concurrency := []int{1, 2, 4, 8, 16}

	for _, c := range concurrency {
		b.Run(fmt.Sprintf("Concurrent_workers%d", c), func(b *testing.B) {
			benchmarkConcurrentOperations(b, c)
		})
	}
}

func benchmarkConcurrentOperations(b *testing.B, numWorkers int) {
	lts, err := NewLinTS(10, WithRandomSeed(42))
	if err != nil {
		b.Fatalf("NewLinTS() error = %v", err)
	}

	// Pre-generate test data
	rng := rand.New(rand.NewSource(42))
	contexts := make([][]float64, 1000)
	rewards := make([]float64, 1000)

	for i := range contexts {
		contexts[i] = make([]float64, 10)
		for j := range contexts[i] {
			contexts[i][j] = rng.NormFloat64()
		}
		rewards[i] = rng.NormFloat64()
	}

	b.ResetTimer()
	b.ReportAllocs()

	var wg sync.WaitGroup
	opsPerWorker := b.N / numWorkers

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			workerRng := rand.New(rand.NewSource(int64(workerID)))

			for i := 0; i < opsPerWorker; i++ {
				idx := workerRng.Intn(len(contexts))

				if workerRng.Float64() < 0.3 { // 30% updates, 70% scores
					lts.Update(contexts[idx], rewards[idx])
				} else {
					lts.Score(contexts[idx])
				}
			}
		}(w)
	}

	wg.Wait()
}

// BenchmarkMemoryUsage tests memory efficiency
func BenchmarkMemoryUsage(b *testing.B) {
	dimensions := []int{10, 50, 100}

	for _, d := range dimensions {
		b.Run(fmt.Sprintf("MemoryEfficiency_d%d", d), func(b *testing.B) {
			benchmarkMemoryEfficiency(b, d)
		})
	}
}

func benchmarkMemoryEfficiency(b *testing.B, d int) {
	lts, err := NewLinTS(d, WithRandomSeed(42))
	if err != nil {
		b.Fatalf("NewLinTS() error = %v", err)
	}

	// Generate test data
	rng := rand.New(rand.NewSource(42))
	context := make([]float64, d)
	for i := range context {
		context[i] = rng.NormFloat64()
	}

	b.ResetTimer()
	b.ReportAllocs()

	// Mix of operations to test memory pooling efficiency
	for i := 0; i < b.N; i++ {
		if i%3 == 0 {
			lts.Update(context, rng.NormFloat64())
		} else {
			lts.Score(context)
		}
	}
}

// BenchmarkCholeskyUpdate tests the efficiency of rank-1 Cholesky updates
func BenchmarkCholeskyUpdate(b *testing.B) {
	dimensions := []int{10, 50, 100, 200}

	for _, d := range dimensions {
		b.Run(fmt.Sprintf("CholeskyUpdate_d%d", d), func(b *testing.B) {
			benchmarkCholeskyUpdate(b, d)
		})
	}
}

func benchmarkCholeskyUpdate(b *testing.B, d int) {
	lts, err := NewLinTS(d, WithRandomSeed(42))
	if err != nil {
		b.Fatalf("NewLinTS() error = %v", err)
	}

	// Pre-warm with some updates to get realistic matrix state
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 10; i++ {
		context := make([]float64, d)
		for j := range context {
			context[j] = rng.NormFloat64()
		}
		lts.Update(context, rng.NormFloat64())
	}

	// Generate test contexts
	contexts := make([][]float64, b.N)
	rewards := make([]float64, b.N)

	for i := 0; i < b.N; i++ {
		contexts[i] = make([]float64, d)
		for j := range contexts[i] {
			contexts[i][j] = rng.NormFloat64()
		}
		rewards[i] = rng.NormFloat64()
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := lts.Update(contexts[i], rewards[i])
		if err != nil {
			b.Fatalf("Update() error = %v", err)
		}
	}
}

// BenchmarkBatchOperations tests batch vs individual operations
func BenchmarkBatchOperations(b *testing.B) {
	batchSizes := []int{1, 10, 50, 100}

	for _, size := range batchSizes {
		b.Run(fmt.Sprintf("BatchScore_size%d", size), func(b *testing.B) {
			benchmarkBatchVsIndividual(b, size)
		})
	}
}

func benchmarkBatchVsIndividual(b *testing.B, batchSize int) {
	lts, err := NewLinTS(10, WithRandomSeed(42))
	if err != nil {
		b.Fatalf("NewLinTS() error = %v", err)
	}

	// Generate test contexts
	rng := rand.New(rand.NewSource(42))
	contexts := make([][]float64, batchSize)
	for i := range contexts {
		contexts[i] = make([]float64, 10)
		for j := range contexts[i] {
			contexts[i][j] = rng.NormFloat64()
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	if batchSize == 1 {
		// Individual scoring
		for i := 0; i < b.N; i++ {
			_, err := lts.Score(contexts[0])
			if err != nil {
				b.Fatalf("Score() error = %v", err)
			}
		}
	} else {
		// Batch scoring
		for i := 0; i < b.N; i++ {
			_, err := lts.ScoreBatch(contexts)
			if err != nil {
				b.Fatalf("ScoreBatch() error = %v", err)
			}
		}
	}
}

// BenchmarkLongRunning tests performance degradation over time
func BenchmarkLongRunning(b *testing.B) {
	lts, err := NewLinTS(20, WithMaintenanceFreq(1000), WithRandomSeed(42))
	if err != nil {
		b.Fatalf("NewLinTS() error = %v", err)
	}

	// Generate test data
	rng := rand.New(rand.NewSource(42))
	context := make([]float64, 20)
	for i := range context {
		context[i] = rng.NormFloat64()
	}

	b.ResetTimer()
	b.ReportAllocs()

	// Simulate long-running scenario with periodic maintenance
	for i := 0; i < b.N; i++ {
		if i%5 == 0 {
			// 20% updates to keep the model learning
			lts.Update(context, rng.NormFloat64())
		} else {
			// 80% scoring operations
			lts.Score(context)
		}
	}
}

// BenchmarkNumericalStability tests performance under numerical stress
func BenchmarkNumericalStability(b *testing.B) {
	lts, err := NewLinTS(15,
		WithMaintenanceFreq(100),
		WithLambda(0.1),
		WithSigma2(0.1),
		WithRandomSeed(42))
	if err != nil {
		b.Fatalf("NewLinTS() error = %v", err)
	}

	// Generate challenging numerical scenarios
	rng := rand.New(rand.NewSource(42))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Create context with varying magnitudes
		context := make([]float64, 15)
		scale := rng.Float64() * 10 // Random scaling factor
		for j := range context {
			context[j] = rng.NormFloat64() * scale
		}

		if i%10 == 0 {
			// Challenging rewards (small and large values)
			reward := rng.NormFloat64() * (1 + rng.Float64()*9)
			lts.Update(context, reward)
		} else {
			lts.Score(context)
		}
	}
}
