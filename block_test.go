package statebrain

import (
	"testing"

	"github.com/unixpickle/weakai/rnn/rnntest"
)

func TestBlock(t *testing.T) {
	block := NewBlock(4, 3)
	rnntest.NewChecker4In(block, block).FullCheck(t)
}
