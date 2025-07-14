package blrts

import (
	"math/rand"
	"runtime"
	"sync"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// BenchmarkSelectAction tests the performance of action selection
func BenchmarkSelectAction(b *testing.B) {
	const (
		nArms      = 100
		contextDim = 20
		alpha      = 1.0
		sigma      = 0.1
		seed       = 42
	)

	blr, err := NewBLRTS(nArms, contextDim, alpha, sigma, seed)
	if err != nil {
		b.Fatalf("Failed to create BLRTS: %v", err)
	}
	context := mat.NewVecDense(contextDim, make([]float64, contextDim))

	// Initialize with some random values
	rng := rand.New(rand.NewSource(123))
	for i := 0; i < contextDim; i++ {
		context.SetVec(i, rng.NormFloat64())
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := blr.SelectAction(context)
		if err != nil {
			b.Fatalf("SelectAction failed: %v", err)
		}
	}
}

// BenchmarkUpdate tests the performance of posterior updates
func BenchmarkUpdate(b *testing.B) {
	const (
		nArms      = 100
		contextDim = 20
		alpha      = 1.0
		sigma      = 0.1
		seed       = 42
	)

	blr, err := NewBLRTS(nArms, contextDim, alpha, sigma, seed)
	if err != nil {
		b.Fatalf("Failed to create BLRTS: %v", err)
	}
	context := mat.NewVecDense(contextDim, make([]float64, contextDim))

	// Initialize with some random values
	rng := rand.New(rand.NewSource(123))
	for i := 0; i < contextDim; i++ {
		context.SetVec(i, rng.NormFloat64())
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		arm := i % nArms
		reward := rng.Float64()
		if err := blr.Update(arm, context, reward); err != nil {
			b.Fatalf("Update failed: %v", err)
		}
	}
}

// BenchmarkSelectAndUpdate tests the combined workflow
func BenchmarkSelectAndUpdate(b *testing.B) {
	const (
		nArms      = 100
		contextDim = 20
		alpha      = 1.0
		sigma      = 0.1
		seed       = 42
	)

	blr, err := NewBLRTS(nArms, contextDim, alpha, sigma, seed)
	if err != nil {
		b.Fatalf("Failed to create BLRTS: %v", err)
	}
	rng := rand.New(rand.NewSource(123))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Generate random context
		contextData := make([]float64, contextDim)
		for j := range contextData {
			contextData[j] = rng.NormFloat64()
		}
		context := mat.NewVecDense(contextDim, contextData)

		// Select action and update
		arm, err := blr.SelectAction(context)
		if err != nil {
			b.Fatalf("SelectAction failed: %v", err)
		}
		reward := rng.Float64()
		if err := blr.Update(arm, context, reward); err != nil {
			b.Fatalf("Update failed: %v", err)
		}
	}
}

// BenchmarkParallelSelectAction tests concurrent action selection
func BenchmarkParallelSelectAction(b *testing.B) {
	const (
		nArms      = 100
		contextDim = 20
		alpha      = 1.0
		sigma      = 0.1
		seed       = 42
		numWorkers = 4
	)

	blr, err := NewBLRTS(nArms, contextDim, alpha, sigma, seed)
	if err != nil {
		b.Fatalf("Failed to create BLRTS: %v", err)
	}

	// Pre-generate contexts to avoid allocation overhead in benchmark
	contexts := make([]*mat.VecDense, numWorkers)
	for i := 0; i < numWorkers; i++ {
		contextData := make([]float64, contextDim)
		rng := rand.New(rand.NewSource(int64(123 + i)))
		for j := range contextData {
			contextData[j] = rng.NormFloat64()
		}
		contexts[i] = mat.NewVecDense(contextDim, contextData)
	}

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		workerID := 0
		for pb.Next() {
			context := contexts[workerID%numWorkers]
			_, err := blr.SelectAction(context)
			if err != nil {
				return // Skip error in benchmark
			}
			workerID++
		}
	})
}

// BenchmarkParallelUpdate tests concurrent updates on different arms
func BenchmarkParallelUpdate(b *testing.B) {
	const (
		nArms      = 100
		contextDim = 20
		alpha      = 1.0
		sigma      = 0.1
		seed       = 42
		numWorkers = 4
	)

	blr, err := NewBLRTS(nArms, contextDim, alpha, sigma, seed)
	if err != nil {
		b.Fatalf("Failed to create BLRTS: %v", err)
	}

	// Pre-generate contexts to avoid allocation overhead
	contexts := make([]*mat.VecDense, numWorkers)
	for i := 0; i < numWorkers; i++ {
		contextData := make([]float64, contextDim)
		rng := rand.New(rand.NewSource(int64(123 + i)))
		for j := range contextData {
			contextData[j] = rng.NormFloat64()
		}
		contexts[i] = mat.NewVecDense(contextDim, contextData)
	}

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		workerID := 0
		rng := rand.New(rand.NewSource(int64(456 + workerID)))

		for pb.Next() {
			context := contexts[workerID%numWorkers]
			arm := workerID % nArms // Different workers update different arms
			reward := rng.Float64()
			if err := blr.Update(arm, context, reward); err != nil {
				return // Skip error in benchmark
			}
			workerID++
		}
	})
}

// BenchmarkScaling tests performance with different problem sizes
func BenchmarkScaling(b *testing.B) {
	sizes := []struct {
		name       string
		nArms      int
		contextDim int
	}{
		{"Small_10x5", 10, 5},
		{"Medium_50x10", 50, 10},
		{"Large_100x20", 100, 20},
		{"XLarge_500x50", 500, 50},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			const (
				alpha = 1.0
				sigma = 0.1
				seed  = 42
			)

			blr, err := NewBLRTS(size.nArms, size.contextDim, alpha, sigma, seed)
			if err != nil {
				b.Fatalf("Failed to create BLRTS: %v", err)
			}
			context := mat.NewVecDense(size.contextDim, make([]float64, size.contextDim))

			rng := rand.New(rand.NewSource(123))
			for i := 0; i < size.contextDim; i++ {
				context.SetVec(i, rng.NormFloat64())
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, err := blr.SelectAction(context)
				if err != nil {
					b.Fatalf("SelectAction failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkMemoryReuse tests the effectiveness of buffer reuse
func BenchmarkMemoryReuse(b *testing.B) {
	const (
		nArms      = 50
		contextDim = 15
		alpha      = 1.0
		sigma      = 0.1
		seed       = 42
	)

	blr, err := NewBLRTS(nArms, contextDim, alpha, sigma, seed)
	if err != nil {
		b.Fatalf("Failed to create BLRTS: %v", err)
	}

	// Use the same context repeatedly to test buffer reuse
	context := mat.NewVecDense(contextDim, make([]float64, contextDim))
	rng := rand.New(rand.NewSource(123))
	for i := 0; i < contextDim; i++ {
		context.SetVec(i, rng.NormFloat64())
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		arm, err := blr.SelectAction(context)
		if err != nil {
			b.Fatalf("SelectAction failed: %v", err)
		}
		reward := rng.Float64()
		if err := blr.Update(arm, context, reward); err != nil {
			b.Fatalf("Update failed: %v", err)
		}
	}
}

// BenchmarkConcurrentWorkload simulates realistic concurrent usage
func BenchmarkConcurrentWorkload(b *testing.B) {
	const (
		nArms      = 100
		contextDim = 20
		alpha      = 1.0
		sigma      = 0.1
		seed       = 42
	)

	blr, err := NewBLRTS(nArms, contextDim, alpha, sigma, seed)
	if err != nil {
		b.Fatalf("Failed to create BLRTS: %v", err)
	}
	numCPU := runtime.NumCPU()

	b.ResetTimer()
	b.ReportAllocs()

	var wg sync.WaitGroup
	workPerGoroutine := b.N / numCPU

	for i := 0; i < numCPU; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(123 + workerID)))

			for j := 0; j < workPerGoroutine; j++ {
				// Generate random context
				contextData := make([]float64, contextDim)
				for k := range contextData {
					contextData[k] = rng.NormFloat64()
				}
				context := mat.NewVecDense(contextDim, contextData)

				// Select and update
				arm, err := blr.SelectAction(context)
				if err != nil {
					return // Skip error in concurrent benchmark
				}
				reward := rng.Float64()
				if err := blr.Update(arm, context, reward); err != nil {
					return // Skip error in concurrent benchmark
				}
			}
		}(i)
	}

	wg.Wait()
}
