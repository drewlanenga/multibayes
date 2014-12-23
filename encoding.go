package multibayes

import (
	"encoding/json"
)

type jsonableClassifier struct {
	Matrix *sparseMatrixInt `json:"matrix"`
}

func (c *UnweightedClassifier) MarshalJSON() ([]byte, error) {
	return json.Marshal(&jsonableClassifier{c.Matrix})
}

func (c *UnweightedClassifier) UnmarshalJSON(buf []byte) error {
	j := jsonableClassifier{}

	err := json.Unmarshal(buf, &j)
	if err != nil {
		return nil
	}

	*c = *NewUnweightedClassifier()
	c.Matrix = j.Matrix

	return nil
}

// Initialize a new classifier from a JSON byte slice.
func NewClassifierFromJSON(buf []byte) (*UnweightedClassifier, error) {
	classifier := &UnweightedClassifier{}

	err := classifier.UnmarshalJSON(buf)
	if err != nil {
		return nil, err
	}

	return classifier, nil
}

func (s *sparseColumnInt) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.Data)
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
