package multibayes

type sparseColumn interface {
	Add(int)
	Count() int
	Expand(int) []float64
}

type sparseColumnInt struct {
	Data []int `json:"data"`
}

type sparseColumnMap struct {
	Data map[int]int `json:"data"`
}

func newSparseColumnInt() *sparseColumnInt {
	return &sparseColumnInt{
		Data: make([]int, 0, 1000),
	}
}

func (s *sparseColumnInt) Add(index int) {
	s.Data = append(s.Data, index)
}

// return the number of rows that contain the column
func (s *sparseColumnInt) Count() int {
	return len(s.Data)
}

// sparse to dense
func (s *sparseColumnInt) Expand(n int) []float64 {
	expanded := make([]float64, n)
	for _, index := range s.Data {
		expanded[index] = 1.0
	}
	return expanded
}

func (s *sparseColumnInt) getData() ([]int, int) {
	return s.Data, len(s.Data)
}

func newSparseColumnMap() *sparseColumnMap {
	return &sparseColumnMap{
		Data: make(map[int]int),
	}
}

func (s *sparseColumnMap) Add(index int) {
	s.Data[index]++
}

func (s *sparseColumnMap) Count() int {
	count := 0
	for _, val := range s.Data {
		count = count + val
	}
	return count
}

func (s *sparseColumnMap) Expand(n int) []float64 {
	expanded := make([]float64, n)
	for index, count := range s.Data {
		expanded[index] = float64(count)
	}
	return expanded
}

func (s *sparseColumnMap) getData() (map[int]int, int) {
	// length != len(s.Data)
	length := 0
	for _, count := range s.Data {
		length += count
	}
	return s.Data, length
}

type sparseMatrix interface {
	Add([]ngram, []string)
	getTokens() map[string]sparseColumn
	getClasses() map[string]sparseColumn
	getN() int
}

type sparseMatrixMap struct {
	Tokens  map[string]*sparseColumnMap `json:"tokens"`  // []map[tokenindex]occurence
	Classes map[string]*sparseColumnInt `json:"classes"` // map[classname]classindex
	N       int                         `json:"n"`       // number of rows currently in the matrix
}

func newSparseMatrixMap() *sparseMatrixMap {
	return &sparseMatrixMap{
		Tokens:  make(map[string]*sparseColumnMap),
		Classes: make(map[string]*sparseColumnInt),
		N:       0,
	}
}

func (s *sparseMatrixMap) Add(ngrams []ngram, classes []string) {
	if len(ngrams) == 0 || len(classes) == 0 {
		return
	}
	for _, class := range classes {
		if _, ok := s.Classes[class]; !ok {
			// sparse column int here
			s.Classes[class] = newSparseColumnInt()
		}

		s.Classes[class].Add(s.N)
	}

	for _, ngram := range ngrams {
		gramString := ngram.String()
		if _, ok := s.Tokens[gramString]; !ok {
			// sparse column map here
			s.Tokens[gramString] = newSparseColumnMap()
		}

		s.Tokens[gramString].Add(s.N)
	}
	// increment the row counter
	s.N++
}

func (s *sparseMatrixMap) getTokens() map[string]*sparseColumnMap {
	return s.Tokens
}

func (s *sparseMatrixMap) getClasses() map[string]*sparseColumnInt {
	return s.Classes
}

func (s *sparseMatrixMap) getN() int {
	return s.N
}

type sparseMatrixInt struct {
	Tokens  map[string]*sparseColumnInt `json:"tokens"`  // []map[tokenindex]occurence
	Classes map[string]*sparseColumnInt `json:"classes"` // map[classname]classindex
	N       int                         `json:"n"`       // number of rows currently in the matrix
}

func newSparseMatrixInt() *sparseMatrixInt {
	return &sparseMatrixInt{
		Tokens:  make(map[string]*sparseColumnInt),
		Classes: make(map[string]*sparseColumnInt),
		N:       0,
	}
}

func (s *sparseMatrixInt) Add(ngrams []ngram, classes []string) {
	if len(ngrams) == 0 || len(classes) == 0 {
		return
	}
	for _, class := range classes {
		if _, ok := s.Classes[class]; !ok {
			// sparse column int here
			s.Classes[class] = newSparseColumnInt()
		}

		s.Classes[class].Add(s.N)
	}

	for _, ngram := range ngrams {
		gramString := ngram.String()
		if _, ok := s.Tokens[gramString]; !ok {
			// sparse column map here
			s.Tokens[gramString] = newSparseColumnInt()
		}

		s.Tokens[gramString].Add(s.N)
	}
	// increment the row counter
	s.N++
}

func (s *sparseMatrixInt) getTokens() map[string]*sparseColumnInt {
	return s.Tokens
}

func (s *sparseMatrixInt) getClasses() map[string]*sparseColumnInt {
	return s.Classes
}

func (s *sparseMatrixInt) getN() int {
	return s.N
}
