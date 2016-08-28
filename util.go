package statebrain

import (
	"math"

	"github.com/unixpickle/num-analysis/linalg"
)

func maxIndex(v linalg.Vector) int {
	maxVal := math.Inf(-1)
	var res int
	for i, x := range v {
		if x >= maxVal {
			maxVal = x
			res = i
		}
	}
	return res
}
