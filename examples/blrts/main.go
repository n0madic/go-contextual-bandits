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
		nArms      = 3   // number of arms (ads/products/treatments)
		contextDim = 4   // context feature dimension
		alpha      = 1.0 // prior precision (regularization)
		sigma      = 0.1 // noise standard deviation
		seed       = 42  // random seed for reproducibility
	)

	bandit, err := blrts.NewBLRTS(nArms, contextDim, alpha, sigma, seed)
	if err != nil {
		log.Fatal(err)
	}

	// Create a separate RNG for simulation to ensure reproducibility
	simRng := rand.New(rand.NewSource(123))

	fmt.Println("=== Learning Phase ===")
	// Simulate contextual bandit learning with FIXED contexts for demonstration
	fixedContexts := [][]float64{
		{1.0, 0.5, -0.3, 0.8},  // Context 1
		{-0.5, 1.2, 0.1, -0.3}, // Context 2
		{0.8, -0.1, 1.0, 0.4},  // Context 3
		{0.2, 0.7, -0.5, 1.1},  // Context 4
		{-0.2, -0.8, 0.6, 0.3}, // Context 5
	}

	for step := 0; step < 20; step++ {
		// Use fixed contexts cyclically for reproducibility
		contextData := fixedContexts[step%len(fixedContexts)]
		context := mat.NewVecDense(contextDim, contextData)

		// Select action using Thompson Sampling
		selectedArm, err := bandit.SelectAction(context)
		if err != nil {
			log.Printf("Error selecting action: %v", err)
			continue
		}

		// Simulate reward based on true underlying model
		// In practice, this would be actual user feedback
		trueReward := simulateReward(contextData, selectedArm, simRng)

		// Update the model with observed reward
		err = bandit.Update(selectedArm, context, trueReward)
		if err != nil {
			log.Printf("Error updating model: %v", err)
			continue
		}

		if step%4 == 0 || step < 5 {
			fmt.Printf("Step %d: Context %v -> Selected arm %d, reward %.3f\n",
				step+1, contextData, selectedArm, trueReward)
		}
	}

	fmt.Println("\n=== Evaluation Phase ===")
	// Test with the same context multiple times to show consistency after learning
	testContext := mat.NewVecDense(contextDim, []float64{1.0, 0.5, -0.3, 0.8})

	fmt.Println("Testing same context 10 times:")
	for i := 0; i < 10; i++ {
		arm, _ := bandit.SelectAction(testContext)
		fmt.Printf("Test %d: Selected arm %d\n", i+1, arm)
	}

	// Evaluate final performance with more samples
	fmt.Println("\nFinal arm selection probabilities (1000 samples):")
	armCounts := make([]int, nArms)
	for i := 0; i < 1000; i++ {
		arm, _ := bandit.SelectAction(testContext)
		armCounts[arm]++
	}

	for arm, count := range armCounts {
		probability := float64(count) / 1000.0
		fmt.Printf("Arm %d: %.1f%%\n", arm, probability*100)
	}

	// Show which arm is actually best for this context
	fmt.Println("\nTrue expected rewards for test context [1.0, 0.5, -0.3, 0.8]:")
	testContextData := []float64{1.0, 0.5, -0.3, 0.8}
	for arm := 0; arm < nArms; arm++ {
		trueExpectedReward := calculateTrueExpectedReward(testContextData, arm)
		fmt.Printf("Arm %d: %.3f\n", arm, trueExpectedReward)
	}
}

// Simulate true reward function (unknown to the algorithm)
func simulateReward(context []float64, arm int, rng *rand.Rand) float64 {
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

	// Add noise using separate RNG
	reward += rng.NormFloat64() * 0.1

	return reward
}

// Calculate true expected reward without noise (for evaluation)
func calculateTrueExpectedReward(context []float64, arm int) float64 {
	trueParams := [][]float64{
		{0.5, -0.3, 0.2, 0.1},  // arm 0 parameters
		{-0.2, 0.8, -0.1, 0.4}, // arm 1 parameters
		{0.1, 0.2, 0.6, -0.2},  // arm 2 parameters
	}

	reward := 0.0
	for i, param := range trueParams[arm] {
		reward += param * context[i]
	}

	return reward
}
