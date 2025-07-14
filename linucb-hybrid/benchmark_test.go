package linucbhybrid

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"gonum.org/v1/gonum/mat"
)

func BenchmarkRecommend(b *testing.B) {
	benchmarks := []struct {
		name          string
		d             int
		m             int
		numCandidates int
	}{
		{"small_2x2_10candidates", 2, 2, 10},
		{"medium_5x3_50candidates", 5, 3, 50},
		{"large_10x5_100candidates", 10, 5, 100},
		{"xlarge_20x10_200candidates", 20, 10, 200},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			l := NewLinUCBHybrid(bm.d, bm.m)
			rng := rand.New(rand.NewSource(42))

			// Generate test data
			userFeat := make([]float64, bm.d)
			for i := range userFeat {
				userFeat[i] = rng.NormFloat64()
			}

			candidates := make(map[string][]float64)
			for i := 0; i < bm.numCandidates; i++ {
				artFeat := make([]float64, bm.d)
				for j := range artFeat {
					artFeat[j] = rng.NormFloat64()
				}
				candidates[fmt.Sprintf("article_%d", i)] = artFeat
			}

			sharedFeatFn := func(user, art []float64) []float64 {
				shared := make([]float64, bm.m)
				for i := 0; i < bm.m && i < len(user) && i < len(art); i++ {
					shared[i] = user[i] * art[i]
				}
				return shared
			}

			// Warm up the algorithm with some updates
			for i := 0; i < 100; i++ {
				artID := fmt.Sprintf("article_%d", i%bm.numCandidates)
				artFeat := candidates[artID]
				reward := rng.Float64()
				l.Update(artID, userFeat, artFeat, sharedFeatFn, reward)
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, err := l.Recommend(userFeat, candidates, sharedFeatFn)
				if err != nil {
					b.Fatalf("Recommend failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkUpdate(b *testing.B) {
	benchmarks := []struct {
		name string
		d    int
		m    int
	}{
		{"small_2x2", 2, 2},
		{"medium_5x3", 5, 3},
		{"large_10x5", 10, 5},
		{"xlarge_20x10", 20, 10},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			l := NewLinUCBHybrid(bm.d, bm.m)
			rng := rand.New(rand.NewSource(42))

			// Pre-generate test data
			userData := make([][]float64, b.N)
			artData := make([][]float64, b.N)
			rewards := make([]float64, b.N)

			for i := 0; i < b.N; i++ {
				userData[i] = make([]float64, bm.d)
				artData[i] = make([]float64, bm.d)
				for j := 0; j < bm.d; j++ {
					userData[i][j] = rng.NormFloat64()
					artData[i][j] = rng.NormFloat64()
				}
				rewards[i] = rng.Float64()
			}

			sharedFeatFn := func(user, art []float64) []float64 {
				shared := make([]float64, bm.m)
				for i := 0; i < bm.m && i < len(user) && i < len(art); i++ {
					shared[i] = user[i] * art[i]
				}
				return shared
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				artID := fmt.Sprintf("article_%d", i%1000) // Cycle through 1000 articles
				err := l.Update(artID, userData[i], artData[i], sharedFeatFn, rewards[i])
				if err != nil {
					b.Fatalf("Update failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkThompsonSample(b *testing.B) {
	benchmarks := []struct {
		name          string
		d             int
		m             int
		numCandidates int
	}{
		{"small_2x2_10candidates", 2, 2, 10},
		{"medium_5x3_50candidates", 5, 3, 50},
		{"large_10x5_100candidates", 10, 5, 100},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			l := NewLinUCBHybrid(bm.d, bm.m)
			rng := rand.New(rand.NewSource(42))

			// Generate test data
			userFeat := make([]float64, bm.d)
			for i := range userFeat {
				userFeat[i] = rng.NormFloat64()
			}

			candidates := make(map[string][]float64)
			for i := 0; i < bm.numCandidates; i++ {
				artFeat := make([]float64, bm.d)
				for j := range artFeat {
					artFeat[j] = rng.NormFloat64()
				}
				candidates[fmt.Sprintf("article_%d", i)] = artFeat
			}

			sharedFeatFn := func(user, art []float64) []float64 {
				shared := make([]float64, bm.m)
				for i := 0; i < bm.m && i < len(user) && i < len(art); i++ {
					shared[i] = user[i] * art[i]
				}
				return shared
			}

			// Warm up the algorithm with some updates
			for i := 0; i < 100; i++ {
				artID := fmt.Sprintf("article_%d", i%bm.numCandidates)
				artFeat := candidates[artID]
				reward := rng.Float64()
				l.Update(artID, userFeat, artFeat, sharedFeatFn, reward)
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, err := l.ThompsonSample(userFeat, candidates, sharedFeatFn, rng)
				if err != nil {
					b.Fatalf("ThompsonSample failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkConcurrentOperations(b *testing.B) {
	l := NewLinUCBHybrid(5, 3, WithMaxArticles(1000))

	userFeat := []float64{1.0, 0.5, 0.3, 0.8, 0.2}
	candidates := make(map[string][]float64)
	for i := 0; i < 50; i++ {
		artFeat := make([]float64, 5)
		for j := range artFeat {
			artFeat[j] = rand.Float64()
		}
		candidates[fmt.Sprintf("article_%d", i)] = artFeat
	}

	sharedFeatFn := func(user, art []float64) []float64 {
		return []float64{user[0] * art[0], user[1] * art[1], user[2] * art[2]}
	}

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		for pb.Next() {
			if rng.Float64() < 0.7 {
				// 70% recommendations
				_, err := l.Recommend(userFeat, candidates, sharedFeatFn)
				if err != nil {
					b.Fatalf("Recommend failed: %v", err)
				}
			} else {
				// 30% updates
				artID := fmt.Sprintf("article_%d", rng.Intn(50))
				artFeat := candidates[artID]
				reward := rng.Float64()
				err := l.Update(artID, userFeat, artFeat, sharedFeatFn, reward)
				if err != nil {
					b.Fatalf("Update failed: %v", err)
				}
			}
		}
	})
}

func BenchmarkMemoryUsage(b *testing.B) {
	l := NewLinUCBHybrid(10, 5, WithMaxArticles(10000))

	userFeat := make([]float64, 10)
	for i := range userFeat {
		userFeat[i] = rand.Float64()
	}

	sharedFeatFn := func(user, art []float64) []float64 {
		shared := make([]float64, 5)
		for i := 0; i < 5; i++ {
			shared[i] = user[i] * art[i]
		}
		return shared
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		artID := fmt.Sprintf("article_%d", i)
		artFeat := make([]float64, 10)
		for j := range artFeat {
			artFeat[j] = rand.Float64()
		}
		reward := rand.Float64()

		err := l.Update(artID, userFeat, artFeat, sharedFeatFn, reward)
		if err != nil {
			b.Fatalf("Update failed: %v", err)
		}

		// Occasionally recommend to test mixed workload
		if i%100 == 0 {
			candidates := map[string][]float64{artID: artFeat}
			_, err := l.Recommend(userFeat, candidates, sharedFeatFn)
			if err != nil {
				b.Fatalf("Recommend failed: %v", err)
			}
		}
	}
}

// BenchmarkLargeScale tests performance with a large number of articles
func BenchmarkLargeScale(b *testing.B) {
	l := NewLinUCBHybrid(20, 10, WithMaxArticles(50000))

	userFeat := make([]float64, 20)
	for i := range userFeat {
		userFeat[i] = rand.Float64()
	}

	// Pre-populate with many articles
	sharedFeatFn := func(user, art []float64) []float64 {
		shared := make([]float64, 10)
		for i := 0; i < 10; i++ {
			shared[i] = user[i] * art[i]
		}
		return shared
	}

	// Warm up with 10000 articles
	for i := 0; i < 10000; i++ {
		artID := fmt.Sprintf("warmup_article_%d", i)
		artFeat := make([]float64, 20)
		for j := range artFeat {
			artFeat[j] = rand.Float64()
		}
		reward := rand.Float64()
		l.Update(artID, userFeat, artFeat, sharedFeatFn, reward)
	}

	// Create large candidate set
	candidates := make(map[string][]float64)
	for i := 0; i < 1000; i++ {
		artFeat := make([]float64, 20)
		for j := range artFeat {
			artFeat[j] = rand.Float64()
		}
		candidates[fmt.Sprintf("candidate_%d", i)] = artFeat
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := l.Recommend(userFeat, candidates, sharedFeatFn)
		if err != nil {
			b.Fatalf("Recommend failed: %v", err)
		}
	}
}

// BenchmarkSamplingMethods compares performance of different sampling approaches
func BenchmarkSamplingMethods(b *testing.B) {
	sizes := []struct {
		name string
		d, m int
	}{
		{"small_3x3", 3, 3},
		{"medium_8x8", 8, 8},
		{"large_16x16", 16, 16},
		{"xlarge_32x32", 32, 32},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			l := NewLinUCBHybrid(size.d, size.m)
			rng := rand.New(rand.NewSource(42))

			// Create article with some realistic data
			aID := "bench_article"
			art := l.getOrCreateArticle(aID)
			defer l.releaseArticle(art, aID)

			// Warm up the article with some updates
			sharedFeatFn := func(user, art []float64) []float64 {
				result := make([]float64, size.m)
				for i := 0; i < size.m; i++ {
					result[i] = user[i%len(user)] * art[i%len(art)]
				}
				return result
			}

			for i := 0; i < 20; i++ {
				userFeat := make([]float64, 2)
				for j := range userFeat {
					userFeat[j] = rand.Float64()
				}
				artFeat := make([]float64, size.d)
				for j := range artFeat {
					artFeat[j] = rand.Float64()
				}
				l.Update(aID, userFeat, artFeat, sharedFeatFn, rand.Float64())
			}

			mean := mat.NewDense(size.d, 1, nil)
			for i := 0; i < size.d; i++ {
				mean.Set(i, 0, rand.Float64())
			}

			// Benchmark direct sampling
			b.Run("direct", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = l.sampleMultivariateNormal(mean, art.AInv, rng)
				}
			})

			// Benchmark optimized sampling
			b.Run("optimized", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = l.sampleMultivariateNormalOptimized(mean, art, rng)
				}
			})
		})
	}
}
