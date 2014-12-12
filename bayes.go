package multibayes

import (
	"math"
)

var (
	smoother     = 1 // laplace
	minClassSize = 5
)

type Classifier struct {
	Tokenizer *tokenizer   `json:"-"`
	Matrix    sparseMatrix `json:"matrix"`
}

// Create a new multibayes classifier.
func NewClassifierMap() *Classifier {
	tokenize, _ := newTokenizer(&tokenizerConf{
		NGramSize: 1,
	})

	sparse := newSparseMatrixMap()

	return &Classifier{
		Tokenizer: tokenize,
		Matrix:    sparse,
	}
}

func NewClassifierInt() *Classifier {
	tokenize, _ := newTokenizer(&tokenizerConf{
		NGramSize: 1,
	})

	sparse := newSparseMatrixInt()

	return &Classifier{
		Tokenizer: tokenize,
		Matrix:    sparse,
	}
}

// Train the classifier with a new document and its classes.
func (c *Classifier) Add(document string, classes []string) {
	ngrams := c.Tokenizer.Parse(document)
	c.Matrix.Add(ngrams, classes)
}

// Calculate the posterior probability for a new document on each
// class from the training set.
func (c *Classifier) Posterior(document string) map[string]float64 {
	tokens := c.Tokenizer.Parse(document)
	predictions := make(map[string]float64)

	for class, classcolumn := range c.Matrix.getClasses() {
		if len(classcolumn.getData()) < minClassSize {
			continue
		}

		n := classcolumn.Count()
		smoothN := n + (smoother * 2)

		priors := []float64{
			float64(n+smoother) / float64(c.Matrix.getN()+(smoother*2)),                 // P(C=Y)
			float64(c.Matrix.getN()-n+smoother) / float64(c.Matrix.getN()+(smoother*2)), // P(C=N)
		}

		loglikelihood := []float64{1.0, 1.0}

		// check if each token is in our token sparse matrix
		for _, token := range tokens {
			if tokencolumn, ok := c.Matrix.getTokens()[token.String()]; ok {
				// conditional probability the token occurs for the class
				joint := intersection(tokencolumn.getData(), classcolumn.getData())
				conditional := float64(joint+smoother) / float64(smoothN) // P(F|C=Y)
				loglikelihood[0] += math.Log(conditional)

				// conditional probability the token occurs if the class doesn't apply
				not := len(tokencolumn.getData()) - joint
				notconditional := float64(not+smoother) / float64(smoothN) // P(F|C=N)
				loglikelihood[1] += math.Log(notconditional)
			}
		}

		likelihood := []float64{
			math.Exp(loglikelihood[0]),
			math.Exp(loglikelihood[1]),
		}

		prob := bayesRule(priors, likelihood) // P(C|F)
		predictions[class] = prob[0]
	}

	return predictions
}

func bayesRule(prior, likelihood []float64) []float64 {

	posterior := make([]float64, len(prior))

	sum := 0.0
	for i, _ := range prior {
		combined := prior[i] * likelihood[i]

		posterior[i] = combined
		sum += combined
	}

	// scale the likelihoods
	for i, _ := range posterior {
		posterior[i] /= sum
	}

	return posterior
}

// define intersection function that takes interfaces
func intersection(sparseColumn1, sparseColumn2 sparseColumn) int {
	var count int
	return count
}

// elements that are in both array1 and array2
func arrayIntersection(array1, array2 []int) int {
	var count int
	for _, elem1 := range array1 {
		for _, elem2 := range array2 {
			if elem1 == elem2 {
				count++
				break
			}
		}
	}
	return count
}

func mapIntersection(map1, map2 map[int]int) int {
	var count int
	for index1, count1 := range map1 {
		for index2, count2 := range map2 {
			if index1 == index2 {
				count = count + (count1 + count2)
			}
		}
	}
	return count
}
