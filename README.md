# Contextual Bandits in Go

[![Go Reference](https://pkg.go.dev/badge/github.com/n0madic/go-contextual-bandits.svg)](https://pkg.go.dev/github.com/n0madic/go-contextual-bandits)

A high-performance implementation of contextual bandit algorithms in Go, optimized for concurrent operations and real-world applications.

## What are Contextual Bandits?

Contextual bandits are a class of online learning algorithms that solve the exploration-exploitation trade-off in sequential decision-making problems. Unlike traditional multi-armed bandits that only consider historical rewards, contextual bandits incorporate additional information (context) about each decision point to make more informed choices.

### Key Concepts

- **Arms/Actions**: Available choices at each decision point
- **Context**: Side information that describes the current situation
- **Reward**: Feedback received after selecting an arm
- **Policy**: Strategy for selecting arms based on context and historical data

### Common Applications

- **Recommendation Systems**: Personalized content, product, or article recommendations
- **Online Advertising**: Ad placement and targeting optimization
- **A/B Testing**: Dynamic allocation of users to test variants
- **Clinical Trials**: Adaptive treatment assignment
- **Resource Allocation**: Dynamic pricing, inventory management
- **Content Optimization**: Website layout, email subject lines
- **Financial Trading**: Portfolio optimization with market context

## Implemented Algorithms

This repository provides three state-of-the-art contextual bandit algorithms, each optimized for different use cases:

### 1. LinUCB Hybrid (`linucb-hybrid/`)

**LinUCB (Linear Upper Confidence Bound) with Hybrid Features** combines user features, item features, and their interactions to make recommendations with confidence bounds.

#### Pros
- **Theoretical Guarantees**: Provable regret bounds under linear reward assumptions
- **Interpretable**: Clear confidence intervals and feature importance
- **Hybrid Features**: Supports both shared and item-specific features
- **Memory Efficient**: LRU caching for large item catalogs
- **Fast Updates**: Sherman-Morrison matrix updates in O(d²) time

#### Cons
- **Linear Assumption**: Assumes linear relationship between context and rewards
- **Cold Start**: Poor performance with completely new items
- **Parameter Tuning**: Requires careful tuning of alpha (exploration parameter)

#### Best Use Cases
- **News/Article Recommendation**: Where content features are important
- **E-commerce**: Product recommendations with user and item features
- **Content Platforms**: When you have rich feature representations
- **Large Item Catalogs**: Thanks to efficient caching mechanism

### 2. Bayesian Linear Regression Thompson Sampling (`bl-rts/`)

**BLR-TS** uses Bayesian inference to maintain posterior distributions over reward parameters and samples from these distributions for exploration.

#### Pros
- **Natural Exploration**: Thompson Sampling provides optimal exploration-exploitation balance
- **Uncertainty Quantification**: Full posterior distributions over parameters
- **Robust to Outliers**: Bayesian framework handles noise well
- **No Hyperparameter Tuning**: Less sensitive to parameter choices than UCB methods
- **Parallel Arms**: Independent updates per arm for better concurrency

#### Cons
- **Computational Cost**: Cholesky decomposition and matrix sampling
- **Memory Usage**: Maintains full covariance matrices per arm
- **Linear Assumption**: Still assumes linear reward model
- **Warm-up Period**: Needs some data to build reliable posteriors

#### Best Use Cases
- **Small to Medium Arm Sets**: Where computational cost is manageable
- **High-Stakes Decisions**: When uncertainty quantification is crucial
- **Noisy Environments**: Robust performance under uncertainty
- **Research/Experimentation**: When interpretability of uncertainty is needed

### 3. Linear Thompson Sampling (`lints/`)

**LinTS** is a streamlined Thompson Sampling implementation with optimizations for numerical stability and performance.

#### Pros
- **High Performance**: O(d²) updates with efficient matrix operations
- **Numerical Stability**: L2-normalized contexts and periodic maintenance
- **Memory Efficient**: Shared parameter sampling across arms
- **Flexible**: Optional bias terms and configurable regularization
- **Thread-Safe**: Optimized for concurrent environments

#### Cons
- **Single Model**: Shares parameters across all arms (less flexible than per-arm models)
- **Linear Assumption**: Limited to linear reward relationships
- **Feature Engineering**: Requires good feature representations

#### Best Use Cases
- **High-Throughput Systems**: When speed is critical
- **Similar Arms**: When arms share similar reward structures
- **Real-Time Applications**: Low-latency decision making
- **Large-Scale Deployment**: Production systems with high QPS

## Algorithm Comparison

| Feature | LinUCB Hybrid | BLR-TS | LinTS |
|---------|---------------|---------|-----------|
| **Exploration Strategy** | UCB | Thompson Sampling | Thompson Sampling |
| **Per-Arm Parameters** | ✅ | ✅ | ❌ |
| **Shared Parameters** | ✅ | ❌ | ✅ |
| **Uncertainty Quantification** | Confidence Bounds | Full Posterior | Posterior Sampling |
| **Computational Complexity** | O(d²+m²) | O(d³) | O(d²) |
| **Memory per Arm** | O(d²+dm) | O(d²) | O(1) |
| **Theoretical Guarantees** | Strong | Strong | Moderate |
| **Parameter Sensitivity** | High | Low | Medium |

## Installation

```bash
go get github.com/n0madic/contextual-bandits
```

## Usage Examples

### LinUCB Hybrid Example

```go
package main

import (
    "fmt"
    "log"
    "math/rand"

    "github.com/n0madic/go-contextual-bandits/linucb-hybrid"
)

func main() {
    // Create LinUCB instance
    // d=3 (article features), m=2 (shared features)
    bandit := linucbhybrid.NewLinUCBHybrid(3, 2,
        linucbhybrid.WithAlpha0(1.0),          // exploration parameter
        linucbhybrid.WithMaxArticles(1000),    // cache size
        linucbhybrid.WithDecay(true),          // decay alpha over time
    )

    // Define a shared feature function (user-article interaction)
    sharedFeatureFunc := func(userFeatures, articleFeatures []float64) []float64 {
        // Simple interaction: [user_age * article_popularity, user_category_match]
        return []float64{
            userFeatures[0] * articleFeatures[2], // age * popularity
            userFeatures[1],                       // category preference
        }
    }

    // Simulate recommendations
    for step := 0; step < 100; step++ {
        // User context (age_normalized, category_preference)
        userFeatures := []float64{rand.Float64(), rand.Float64()}

        // Available articles with features (relevance, quality, popularity)
        candidates := map[string][]float64{
            "article_1": {0.8, 0.9, 0.7},
            "article_2": {0.6, 0.8, 0.9},
            "article_3": {0.9, 0.7, 0.6},
        }

        // Get best recommendation
        bestArticle, err := bandit.Recommend(userFeatures, candidates, sharedFeatureFunc)
        if err != nil {
            log.Printf("Error getting recommendation: %v", err)
            continue
        }

        if bestArticle == "" {
            log.Printf("No recommendation returned")
            continue
        }

        fmt.Printf("Step %d: Recommended %s\n", step+1, bestArticle)

        // Simulate user feedback (1 for click, 0 for no click)
        reward := 0.0
        if rand.Float64() < 0.3 { // 30% click rate
            reward = 1.0
        }

        // Update the model with observed reward
        bestArticleFeatures := candidates[bestArticle]
        err = bandit.Update(bestArticle, userFeatures, bestArticleFeatures, sharedFeatureFunc, reward)
        if err != nil {
            log.Printf("Error updating model: %v", err)
        }
    }
}
```

### BLR-TS Example

```go
package main

import (
    "fmt"
    "log"
    "math/rand"

    blrts "github.com/n0madic/go-contextual-bandits/blr-ts"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Create BLR-TS instance
    const (
        nArms      = 3    // number of arms (ads/products/treatments)
        contextDim = 4    // context feature dimension
        alpha      = 1.0  // prior precision (regularization)
        sigma      = 0.1  // noise standard deviation
        seed       = 42   // random seed for reproducibility
    )

    bandit, err := blrts.NewBLRTS(nArms, contextDim, alpha, sigma, seed)
    if err != nil {
        log.Fatal(err)
    }

    // Simulate contextual bandit learning
    for step := 0; step < 200; step++ {
        // Generate random context (user features, time, etc.)
        contextData := make([]float64, contextDim)
        for i := range contextData {
            contextData[i] = rand.NormFloat64()
        }
        context := mat.NewVecDense(contextDim, contextData)

        // Select action using Thompson Sampling
        selectedArm, err := bandit.SelectAction(context)
        if err != nil {
            log.Printf("Error selecting action: %v", err)
            continue
        }

        // Simulate reward based on true underlying model
        // In practice, this would be actual user feedback
        trueReward := simulateReward(contextData, selectedArm)

        // Update the model with observed reward
        err = bandit.Update(selectedArm, context, trueReward)
        if err != nil {
            log.Printf("Error updating model: %v", err)
            continue
        }

        if step%20 == 0 {
            fmt.Printf("Step %d: Selected arm %d, reward %.3f\n",
                step+1, selectedArm, trueReward)
        }
    }

    // Evaluate final performance
    fmt.Println("\nFinal arm selection probabilities:")
    testContextData := []float64{1.0, 0.5, -0.3, 0.8}
    testContext := mat.NewVecDense(contextDim, testContextData)

    armCounts := make([]int, nArms)
    for i := 0; i < 1000; i++ {
        arm, _ := bandit.SelectAction(testContext)
        armCounts[arm]++
    }

    for arm, count := range armCounts {
        probability := float64(count) / 1000.0
        fmt.Printf("Arm %d: %.1f%%\n", arm, probability*100)
    }
}

// Simulate true reward function (unknown to the algorithm)
func simulateReward(context []float64, arm int) float64 {
    // Different true parameters for each arm
    trueParams := [][]float64{
        {0.5, -0.3, 0.2, 0.1},  // arm 0 parameters
        {-0.2, 0.8, -0.1, 0.4}, // arm 1 parameters
        {0.1, 0.2, 0.6, -0.2},  // arm 2 parameters
    }

    reward := 0.0
    for i, param := range trueParams[arm] {
        reward += param * context[i]
    }

    // Add noise
    reward += rand.NormFloat64() * 0.1

    return reward
}
```

### LinTS Example

```go
package main

import (
    "fmt"
    "log"
    "math/rand"

    "github.com/n0madic/go-contextual-bandits/lints"
)

func main() {
    // Create Linear TS instance
    const dFeatures = 5 // feature dimension

    bandit, err := lints.NewLinTS(dFeatures,
        lints.WithLambda(1.0),           // regularization
        lints.WithSigma2(0.25),          // noise variance
        lints.WithBias(true),            // include bias term
        lints.WithMaintenanceFreq(100),  // numerical maintenance
        lints.WithRandomSeed(42),        // reproducible results
    )
    if err != nil {
        log.Fatal(err)
    }

    // Available actions (e.g., different ad creatives, treatments)
    actions := []struct {
        id       string
        features []float64
    }{
        {"action_A", []float64{1.0, 0.5, -0.2, 0.8, 0.3}},
        {"action_B", []float64{-0.5, 1.2, 0.1, -0.3, 0.9}},
        {"action_C", []float64{0.8, -0.1, 1.0, 0.4, -0.6}},
        {"action_D", []float64{0.2, 0.7, -0.5, 1.1, 0.1}},
    }

    // Simulate online learning
    for step := 0; step < 300; step++ {
        // Score all actions
        scores := make([]float64, len(actions))
        for i, action := range actions {
            score, err := bandit.Score(action.features)
            if err != nil {
                log.Printf("Error scoring action %s: %v", action.id, err)
                continue
            }
            scores[i] = score
        }

        // Select action with highest score (Thompson Sampling)
        bestIdx := 0
        bestScore := scores[0]
        for i := 1; i < len(scores); i++ {
            if scores[i] > bestScore {
                bestScore = scores[i]
                bestIdx = i
            }
        }

        selectedAction := actions[bestIdx]

        // Simulate reward (in practice, this would be user feedback)
        trueReward := simulateLinearReward(selectedAction.features)

        // Update model with observed reward
        err = bandit.Update(selectedAction.features, trueReward)
        if err != nil {
            log.Printf("Error updating model: %v", err)
            continue
        }

        if step%30 == 0 {
            fmt.Printf("Step %d: Selected %s (score: %.3f, reward: %.3f)\n",
                step+1, selectedAction.id, bestScore, trueReward)
        }
    }

    // Final evaluation
    fmt.Println("\nFinal action preferences:")
    for _, action := range actions {
        score, _ := bandit.Score(action.features)
        fmt.Printf("%s: %.3f\n", action.id, score)
    }

    // Access learned parameters through the underlying structure
    // Note: In production, you'd typically use Save/Load for model persistence
    fmt.Printf("\nModel has been trained successfully\n")
}

// Simulate true linear reward function
func simulateLinearReward(features []float64) float64 {
    // True parameters (unknown to the algorithm)
    trueParams := []float64{0.3, -0.8, 0.5, 0.2, -0.4}
    trueBias := 0.1

    reward := trueBias
    for i, param := range trueParams {
        if i < len(features) {
            reward += param * features[i]
        }
    }

    // Add noise
    reward += rand.NormFloat64() * 0.1

    return reward
}
```

## Advanced Features

### Serialization and Persistence

All algorithms support serialization for model persistence:

```go
// Save model state
var buf bytes.Buffer
err := bandit.Save(&buf)
if err != nil {
    log.Fatal(err)
}

// Load model state
err = bandit.Load(&buf)
if err != nil {
    log.Fatal(err)
}
```

### Concurrent Operations

All implementations are thread-safe and optimized for concurrent use:

```go
// Safe to call from multiple goroutines
go func() {
    score, _ := bandit.Recommend(userFeatures, itemFeatures, itemID, sharedFunc)
    // Process recommendation...
}()

go func() {
    bandit.Update(userFeatures, itemFeatures, itemID, reward, sharedFunc)
    // Handle update...
}()
```

### Performance Optimization

- **Memory Management**: Pre-allocated buffers and object pooling
- **Matrix Operations**: Efficient linear algebra using gonum
- **Cache-Friendly**: LRU caching for large item catalogs (LinUCB)
- **Lock Granularity**: Fine-grained locking for better parallelism

## Dependencies

- [gonum](https://gonum.org/): Numerical computing library for linear algebra operations

## Testing

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run benchmarks
go test -bench=. -benchmem ./...

# Run specific package tests
go test -v ./linucb-hybrid
go test -v ./blr-ts
go test -v ./lints
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
