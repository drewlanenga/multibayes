package multibayes

import (
	"github.com/bmizerany/assert"
	"testing"
)

func TestUnweightedClassifierJSON(t *testing.T) {
	classifier := NewUnweightedClassifier()
	classifier.trainWithTestData()

	b, err := classifier.MarshalJSON()
	assert.Equalf(t, nil, err, "Error marshaling JSON: %v\n", err)

	newclass, err := NewUnweightedClassifierFromJSON(b)
	assert.Equalf(t, nil, err, "Error unmarshaling JSON: %v\n", err)

	assert.Equalf(t, 5, len(newclass.Matrix.Tokens), "Incorrect token length")
	assert.Equalf(t, 2, len(newclass.Matrix.Classes), "Incorrect class length")
}

func TestWeightedClassifierJSON(t *testing.T) {
	classifier := NewWeightedClassifier()
	classifier.trainwithTestData()

	b, err := classifier.MarshalJSON()
	assert.Equal(t, nil, err, "Error marshaling JSON: %v\n", err)

	newclass, err := NewWeightedClassifierFromJSON(b)
	assert.Equalf(t, nil, err, "Error unmarshaling JSON: %v\n", err)

	assert.Equalf(t, 5, len(newclass.Matrix.Tokens), "Incorrect token length")
	assert.Equalf(t, 2, len(newclass.Matrix.Classes), "Incorrect class length")
}
