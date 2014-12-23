package multibayes

import (
	//"bytes"
	"fmt"
	"math"
	//"reflect"
)

var (
	smoother     = 1 // laplace
	minClassSize = 2
)

type Classifier interface {
	Add(string, []string)
	Posterior(string) map[string]float64
}

type WeightedClassifier struct {
	Tokenizer *tokenizer
	Matrix    *sparseMatrixMap
}

type UnweightedClassifier struct {
	Tokenizer *tokenizer
	Matrix    *sparseMatrixInt
}

// Create a new multibayes classifier.
func NewWeightedClassifier() *WeightedClassifier {
	tokenize, _ := newTokenizer(&tokenizerConf{
		NGramSize: 1,
	})

	sparse := newSparseMatrixMap()

	return &WeightedClassifier{
		Tokenizer: tokenize,
		Matrix:    sparse,
	}
}

func NewUnweightedClassifier() *UnweightedClassifier {
	tokenize, _ := newTokenizer(&tokenizerConf{
		NGramSize: 1,
	})

	sparse := newSparseMatrixInt()

	return &UnweightedClassifier{
		Tokenizer: tokenize,
		Matrix:    sparse,
	}
}

// Train the classifier with a new document and its classes.
func (c *WeightedClassifier) Add(document string, classes []string) {
	ngrams := c.Tokenizer.Parse(document)
	// want all elements here:
	c.Matrix.Add(ngrams, classes)
}

func (c *UnweightedClassifier) Add(document string, classes []string) {
	ngrams := c.Tokenizer.Parse(document)
	// want the unique elements here:
	var ngramsuniq []ngram
	var exclude bool
	for _, ngram := range ngrams {
		exclude = false
		for _, ngramuniq := range ngramsuniq {
			if ngram.String() == ngramuniq.String() {
				exclude = true
			}
		}
		if !exclude {
			ngramsuniq = append(ngramsuniq, ngram)
		}
	}
	c.Matrix.Add(ngramsuniq, classes)
}

// Calculate the posterior probability for a new document on each
// class from the training set.
func (c *WeightedClassifier) Posterior(document string) map[string]float64 {
	tokens := c.Tokenizer.Parse(document)
	predictions := make(map[string]float64)

	for class, classcolumn := range c.Matrix.getClasses() {
		fmt.Printf("Class: %v\t Column: %v\n", class, classcolumn)
		classdata, classlength := classcolumn.getData()
		if classlength < minClassSize {
			fmt.Printf("Class length: %v\n", classlength)
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
				fmt.Printf("Token: %v\n", token.String())
				//fmt.Printf("Token column: %+v\n", tokencolumn)
				tokendata, tokenlength := tokencolumn.getData()
				// conditional probability the token occurs for the class
				joint := mapIntersection(tokendata, classdata)
				fmt.Printf("Tokendata: %v\t TokendataType: %T\t Classdata: %v\t ClassdataType: %T\t Intersection: %v\n", tokendata, tokendata, classdata, classdata, joint)
				conditional := float64(joint+smoother) / float64(smoothN) // P(F|C=Y)
				fmt.Printf("Conditional: %v\n", conditional)
				loglikelihood[0] += math.Log(conditional)

				// conditional probability the token occurs if the class doesn't apply
				not := tokenlength - joint
				notconditional := float64(not+smoother) / float64(smoothN) // P(F|C=N)
				fmt.Printf("NotConditional: %v\n", notconditional)
				loglikelihood[1] += math.Log(notconditional)
			}
		}

		likelihood := []float64{
			math.Exp(loglikelihood[0]),
			math.Exp(loglikelihood[1]),
		}

		fmt.Printf("Priors: %v\t Likelihood: %v\n", priors, likelihood)
		prob := bayesRule(priors, likelihood) // P(C|F)
		predictions[class] = prob[0]
		fmt.Println("----")
	}

	return predictions
}

func (c *UnweightedClassifier) Posterior(document string) map[string]float64 {
	tokens := c.Tokenizer.Parse(document)
	predictions := make(map[string]float64)

	for class, classcolumn := range c.Matrix.getClasses() {
		fmt.Printf("Class: %v\t Column: %v\n", class, classcolumn)
		classdata, classlength := classcolumn.getData()
		if classlength < minClassSize {
			fmt.Printf("Class length: %v\n", classlength)
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
				fmt.Printf("Token: %v\n", token.String())
				tokendata, tokenlength := tokencolumn.getData()
				// conditional probability the token occurs for the class
				joint := arrayIntersection(tokendata, classdata)
				fmt.Printf("Tokendata: %v\t TokendataType: %T\t Classdata: %v\t ClassdataType: %T\t Intersection: %v\n", tokendata, tokendata, classdata, classdata, joint)
				conditional := float64(joint+smoother) / float64(smoothN) // P(F|C=Y)
				loglikelihood[0] += math.Log(conditional)

				// conditional probability the token occurs if the class doesn't apply
				not := tokenlength - joint
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
		fmt.Println("----")
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

func mapIntersection(map1 map[int]int, array1 []int) int {
	var count int
	for index1, count1 := range map1 {
		for _, index2 := range array1 {
			if index1 == index2 {
				count += count1
			}
		}
	}
	return count
}
