package multibayes

import (
	"encoding/json"
	"strconv"
)

type jsonableUnweightedClassifier struct {
	Matrix *sparseMatrixInt `json:"matrix"`
}

type jsonableWeightedClassifier struct {
	Matrix *sparseMatrixMap `json:"matrix"`
}

func (c *UnweightedClassifier) MarshalJSON() ([]byte, error) {
	return json.Marshal(&jsonableUnweightedClassifier{c.Matrix})
}

func (c *WeightedClassifier) MarshalJSON() ([]byte, error) {
	return json.Marshal(&jsonableWeightedClassifier{c.Matrix})
}

func (c *UnweightedClassifier) UnmarshalJSON(buf []byte) error {
	j := jsonableUnweightedClassifier{}

	err := json.Unmarshal(buf, &j)
	if err != nil {
		return nil
	}

	*c = *NewUnweightedClassifier()
	c.Matrix = j.Matrix

	return nil
}

func (c *WeightedClassifier) UnmarshalJSON(buf []byte) error {
	j := jsonableWeightedClassifier{}

	err := json.Unmarshal(buf, &j)
	if err != nil {
		return nil
	}

	*c = *NewWeightedClassifier()
	c.Matrix = j.Matrix

	return nil
}

// Initialize a new classifier from a JSON byte slice.
func NewUnweightedClassifierFromJSON(buf []byte) (*UnweightedClassifier, error) {
	classifier := &UnweightedClassifier{}

	err := classifier.UnmarshalJSON(buf)
	if err != nil {
		return nil, err
	}

	return classifier, nil
}

func NewWeightedClassifierFromJSON(buf []byte) (*WeightedClassifier, error) {
	classifier := &WeightedClassifier{}

	err := classifier.UnmarshalJSON(buf)
	if err != nil {
		return nil, err
	}

	return classifier, nil
}

func (s *sparseColumnInt) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.Data)
}

func (s *sparseColumnMap) MarshalJSON() ([]byte, error) {
	// convert key values from ints to strings for encoding
	jsonableMap := make(map[string]int)
	for key, val := range s.Data {
		jsonableMap[strconv.Itoa(key)] = val
	}
	return json.Marshal(jsonableMap)
}

func (s *sparseColumnInt) UnmarshalJSON(buf []byte) error {
	var data []int

	err := json.Unmarshal(buf, &data)
	if err != nil {
		return err
	}

	s.Data = data

	return nil
}

func (s *sparseColumnMap) UnmarshalJSON(buf []byte) error {
	jsonableData := make(map[string]int)

	err := json.Unmarshal(buf, &jsonableData)
	if err != nil {
		return err
	}

	// convert map[string]int to map[int]int
	data := make(map[int]int)
	for key, val := range jsonableData {
		i, err := strconv.Atoi(key)
		if err != nil {
			continue
		}
		data[i] = val
	}
	s.Data = data

	return nil
}
