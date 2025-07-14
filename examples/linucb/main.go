package main

import (
	"fmt"
	"log"
	"math/rand"

	linucbhybrid "github.com/n0madic/go-contextual-bandits/linucb-hybrid"
)

func main() {
	// Create LinUCB Hybrid instance
	const (
		dFeatures = 4 // article feature dimension
		mShared   = 3 // shared feature dimension
	)

	bandit := linucbhybrid.NewLinUCBHybrid(dFeatures, mShared,
		linucbhybrid.WithAlpha0(1.0),       // exploration parameter
		linucbhybrid.WithMaxArticles(1000), // cache size for articles
		linucbhybrid.WithDecay(true),       // decay alpha over time
		linucbhybrid.WithClipHigh(10.0),    // clip confidence bonus
		linucbhybrid.WithEps(1e-8),         // numerical stability
	)

	// Create a separate RNG for simulation to ensure reproducibility
	simRng := rand.New(rand.NewSource(123))

	// Define articles (e.g., different news articles, ads, products)
	articles := []struct {
		id       string
		features []float64 // [category_score, quality, freshness, popularity]
	}{
		{"tech_news", []float64{1.0, 0.8, 0.9, 0.7}},     // tech category, high quality
		{"sports_news", []float64{0.2, 0.9, 0.8, 0.9}},   // sports category, very popular
		{"politics_news", []float64{0.5, 0.7, 1.0, 0.6}}, // politics, fresh content
		{"health_news", []float64{0.8, 0.9, 0.6, 0.5}},   // health, high quality
	}

	// Fixed user contexts for reproducible demonstration
	fixedUsers := []struct {
		id       string
		features []float64 // [tech_interest, engagement_history]
	}{
		{"tech_user", []float64{0.9, 0.8}},   // tech-interested, highly engaged
		{"casual_user", []float64{0.3, 0.4}}, // low tech interest, casual engagement
		{"sports_fan", []float64{0.1, 0.9}},  // not tech-interested, highly engaged
		{"news_reader", []float64{0.6, 0.7}}, // moderate tech interest, good engagement
	}

	// Define shared feature function (user-article interaction)
	sharedFeatureFunc := func(userFeatures, articleFeatures []float64) []float64 {
		// Interaction features combining user preferences with article properties
		return []float64{
			userFeatures[0] * articleFeatures[0], // tech_interest * category_score
			userFeatures[1] * articleFeatures[1], // engagement * quality
			userFeatures[1] * articleFeatures[3], // engagement * popularity
		}
	}

	fmt.Println("=== Learning Phase ===")
	// Simulate online learning with deterministic progression
	for step := 0; step < 40; step++ {
		// Use fixed user contexts cyclically for reproducibility
		user := fixedUsers[step%len(fixedUsers)]

		// Prepare candidates map
		candidates := make(map[string][]float64)
		for _, article := range articles {
			candidates[article.id] = article.features
		}

		// Get recommendation using upper confidence bound
		selectedArticle, err := bandit.Recommend(user.features, candidates, sharedFeatureFunc)
		if err != nil {
			log.Printf("Error getting recommendation: %v", err)
			continue
		}

		if selectedArticle == "" {
			log.Printf("No recommendation returned")
			continue
		}

		// Find selected article features
		var selectedFeatures []float64
		for _, article := range articles {
			if article.id == selectedArticle {
				selectedFeatures = article.features
				break
			}
		}

		// Simulate reward based on true user preferences
		trueReward := simulateReward(user.features, selectedFeatures, user.id, selectedArticle, simRng)

		// Update model with observed reward
		err = bandit.Update(selectedArticle, user.features, selectedFeatures, sharedFeatureFunc, trueReward)
		if err != nil {
			log.Printf("Error updating model: %v", err)
			continue
		}

		if step%8 == 0 || step < 5 {
			fmt.Printf("Step %d: User %s -> Recommended %s (reward: %.3f)\n",
				step+1, user.id, selectedArticle, trueReward)
		}
	}

	fmt.Println("\n=== Evaluation Phase ===")
	// Test with a specific user multiple times to show Thompson Sampling variability
	testUser := fixedUsers[0] // tech_user
	testRng := rand.New(rand.NewSource(456))

	candidates := make(map[string][]float64)
	for _, article := range articles {
		candidates[article.id] = article.features
	}

	fmt.Printf("Thompson Sampling for %s (10 samples):\n", testUser.id)
	for i := 0; i < 10; i++ {
		selectedArticle, err := bandit.ThompsonSample(testUser.features, candidates, sharedFeatureFunc, testRng)
		if err != nil {
			log.Printf("Error in Thompson sampling: %v", err)
			continue
		}
		fmt.Printf("Sample %d: %s\n", i+1, selectedArticle)
	}

	// Final evaluation: recommendation frequencies
	fmt.Printf("\nRecommendation frequencies for %s (1000 samples):\n", testUser.id)
	articleCounts := make(map[string]int)
	evalRng := rand.New(rand.NewSource(789))

	for i := 0; i < 1000; i++ {
		selectedArticle, err := bandit.ThompsonSample(testUser.features, candidates, sharedFeatureFunc, evalRng)
		if err != nil {
			continue
		}
		articleCounts[selectedArticle]++
	}

	for _, article := range articles {
		count := articleCounts[article.id]
		percentage := float64(count) / 10.0 // convert to percentage
		fmt.Printf("%s: %.1f%% (%d times)\n", article.id, percentage, count)
	}

	// Show true expected rewards for comparison
	fmt.Printf("\nTrue expected rewards for %s:\n", testUser.id)
	for _, article := range articles {
		trueExpectedReward := calculateTrueExpectedReward(testUser.features, article.features, testUser.id, article.id)
		fmt.Printf("%s: %.3f\n", article.id, trueExpectedReward)
	}

	fmt.Printf("\nLinUCB Hybrid model has been trained successfully!\n")
}

// Simulate true reward function (unknown to the algorithm)
func simulateReward(userFeatures, articleFeatures []float64, userID, articleID string, rng *rand.Rand) float64 {
	// Base reward from feature interaction
	reward := 0.1 // base reward

	// User tech interest * article category relevance
	reward += userFeatures[0] * articleFeatures[0] * 0.5

	// User engagement * article quality
	reward += userFeatures[1] * articleFeatures[1] * 0.3

	// Article freshness bonus
	reward += articleFeatures[2] * 0.2

	// User-specific preferences (unknown to algorithm)
	switch userID {
	case "tech_user":
		if articleID == "tech_news" {
			reward += 0.4 // strong preference for tech news
		}
	case "sports_fan":
		if articleID == "sports_news" {
			reward += 0.5 // strong preference for sports
		}
	case "news_reader":
		if articleID == "politics_news" {
			reward += 0.3 // moderate preference for politics
		}
	case "casual_user":
		// Casual users prefer popular content
		reward += articleFeatures[3] * 0.2
	}

	// Add noise using separate RNG
	reward += rng.NormFloat64() * 0.1

	// Ensure reward is in reasonable range
	if reward < 0 {
		reward = 0
	}
	if reward > 1 {
		reward = 1
	}

	return reward
}

// Calculate true expected reward without noise (for evaluation)
func calculateTrueExpectedReward(userFeatures, articleFeatures []float64, userID, articleID string) float64 {
	reward := 0.1 // base reward

	// Feature interactions
	reward += userFeatures[0] * articleFeatures[0] * 0.5
	reward += userFeatures[1] * articleFeatures[1] * 0.3
	reward += articleFeatures[2] * 0.2

	// User-specific preferences
	switch userID {
	case "tech_user":
		if articleID == "tech_news" {
			reward += 0.4
		}
	case "sports_fan":
		if articleID == "sports_news" {
			reward += 0.5
		}
	case "news_reader":
		if articleID == "politics_news" {
			reward += 0.3
		}
	case "casual_user":
		reward += articleFeatures[3] * 0.2
	}

	if reward < 0 {
		reward = 0
	}
	if reward > 1 {
		reward = 1
	}

	return reward
}
