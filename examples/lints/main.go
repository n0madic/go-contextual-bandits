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
		lints.WithLambda(1.0),          // regularization
		lints.WithSigma2(0.25),         // noise variance
		lints.WithBias(true),           // include bias term
		lints.WithMaintenanceFreq(100), // numerical maintenance
		lints.WithRandomSeed(42),       // reproducible results
	)
	if err != nil {
		log.Fatal(err)
	}

	// Create a separate RNG for simulation to ensure reproducibility
	simRng := rand.New(rand.NewSource(123))

	// Available actions (e.g., different ad creatives, treatments)
	actions := []struct {
		id       string
		features []float64
	}{
		{"premium_ad", []float64{1.0, 0.5, -0.2, 0.8, 0.3}},   // premium ad creative
		{"budget_ad", []float64{-0.5, 1.2, 0.1, -0.3, 0.9}},   // budget-friendly ad
		{"trending_ad", []float64{0.8, -0.1, 1.0, 0.4, -0.6}}, // trending topic ad
		{"classic_ad", []float64{0.2, 0.7, -0.5, 1.1, 0.1}},   // classic approach ad
	}

	fmt.Println("=== Learning Phase ===")
	// Simulate online learning with deterministic progression
	for step := 0; step < 50; step++ {
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
		trueReward := simulateLinearReward(selectedAction.features, selectedAction.id, simRng)

		// Update model with observed reward
		err = bandit.Update(selectedAction.features, trueReward)
		if err != nil {
			log.Printf("Error updating model: %v", err)
			continue
		}

		if step%10 == 0 || step < 5 {
			fmt.Printf("Step %d: Selected %s (score: %.3f, reward: %.3f)\n",
				step+1, selectedAction.id, bestScore, trueReward)
		}
	}

	fmt.Println("\n=== Evaluation Phase ===")
	// Test consistency by scoring the same action multiple times
	testAction := actions[0] // premium_ad
	fmt.Printf("Testing %s action 10 times:\n", testAction.id)
	for i := 0; i < 10; i++ {
		score, _ := bandit.Score(testAction.features)
		fmt.Printf("Test %d: Score %.3f\n", i+1, score)
	}

	// Final evaluation of all actions
	fmt.Println("\nFinal action preferences (average of 100 scores):")
	for _, action := range actions {
		totalScore := 0.0
		for i := 0; i < 100; i++ {
			score, _ := bandit.Score(action.features)
			totalScore += score
		}
		avgScore := totalScore / 100.0
		fmt.Printf("%s: %.3f\n", action.id, avgScore)
	}

	// Show true expected rewards
	fmt.Println("\nTrue expected rewards:")
	for _, action := range actions {
		trueExpectedReward := calculateTrueLinearReward(action.features, action.id)
		fmt.Printf("%s: %.3f\n", action.id, trueExpectedReward)
	}

	fmt.Printf("\nModel has been trained successfully\n")
}

// Simulate true linear reward function with action-specific effects
func simulateLinearReward(features []float64, actionID string, rng *rand.Rand) float64 {
	// True parameters (unknown to the algorithm)
	trueParams := []float64{0.3, -0.8, 0.5, 0.2, -0.4}
	trueBias := 0.1

	reward := trueBias
	for i, param := range trueParams {
		if i < len(features) {
			reward += param * features[i]
		}
	}

	// Add action-specific bonuses (unknown to algorithm)
	switch actionID {
	case "premium_ad":
		reward += 0.2 // premium ads perform slightly better
	case "trending_ad":
		reward += 0.1 // trending ads get small boost
	case "budget_ad":
		reward -= 0.1 // budget ads perform slightly worse
		// classic_ad gets no modifier
	}

	// Add noise using separate RNG
	reward += rng.NormFloat64() * 0.1

	return reward
}

// Calculate true expected reward without noise (for evaluation)
func calculateTrueLinearReward(features []float64, actionID string) float64 {
	trueParams := []float64{0.3, -0.8, 0.5, 0.2, -0.4}
	trueBias := 0.1

	reward := trueBias
	for i, param := range trueParams {
		if i < len(features) {
			reward += param * features[i]
		}
	}

	switch actionID {
	case "premium_ad":
		reward += 0.2
	case "trending_ad":
		reward += 0.1
	case "budget_ad":
		reward -= 0.1
	}

	return reward
}
