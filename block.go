package statebrain

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

// A StateEntry represents one fuzzy Markov state in
// a bigger statebrain model.
type StateEntry struct {
	Output      *autofunc.Variable
	Transitions []*autofunc.Variable
}

// A Block is an rnn.Block for the statebrain model.
type Block struct {
	Entries []StateEntry
}

// NewBlock creates a Block with the given alphabet size
// and number of states.
func NewBlock(alphabetSize, stateCount int) *Block {
	res := &Block{Entries: make([]StateEntry, stateCount)}
	for i := range res.Entries {
		res.Entries[i].Output = &autofunc.Variable{
			Vector: make(linalg.Vector, alphabetSize),
		}
		res.Entries[i].Transitions = make([]*autofunc.Variable, alphabetSize)
		for j := range res.Entries[i].Transitions {
			res.Entries[i].Transitions[j] = &autofunc.Variable{
				Vector: make(linalg.Vector, stateCount),
			}
		}
	}
	return res
}

// StateSize returns the number of states.
func (b *Block) StateSize() int {
	return len(b.Entries)
}

// Batch applies the block to a batch of input vectors.
func (b *Block) Batch(in *rnn.BlockInput) rnn.BlockOutput {
	out := &blockOutput{
		OutputRes:  make([]autofunc.Result, len(in.Inputs)),
		StateRes:   make([]autofunc.Result, len(in.Inputs)),
		OutputVecs: make([]linalg.Vector, len(in.Inputs)),
		StateVecs:  make([]linalg.Vector, len(in.Inputs)),
	}

	var softmax autofunc.Softmax
	for i, inputVec := range in.Inputs {
		inSoftmax := softmax.Apply(inputVec)
		var input int
		var maxValue float64
		for j, x := range inSoftmax.Output() {
			if x >= maxValue {
				maxValue = x
				input = j
			}
		}

		stateProbs := softmax.Apply(in.States[i])
		res := autofunc.Pool(stateProbs, func(inState autofunc.Result) autofunc.Result {
			var output autofunc.Result
			var newStates autofunc.Result
			for stateIdx, entry := range b.Entries {
				prob := autofunc.Slice(inState, stateIdx, stateIdx+1)
				scaledOut := autofunc.ScaleFirst(entry.Output, prob)
				scaledStates := autofunc.ScaleFirst(entry.Transitions[input], prob)
				if output == nil {
					output = scaledOut
					newStates = scaledStates
				} else {
					output = autofunc.Add(output, scaledOut)
					newStates = autofunc.Add(newStates, scaledOut)
				}
			}
			return autofunc.Concat(output, newStates)
		})
		out.OutputRes[i] = autofunc.Slice(res, 0, len(inputVec.Output()))
		out.OutputVecs[i] = out.OutputRes[i].Output()
		out.StateRes[i] = autofunc.Slice(res, len(inputVec.Output()), len(res.Output()))
		out.StateVecs[i] = out.StateRes[i].Output()
	}

	return out
}

type blockOutput struct {
	OutputRes  []autofunc.Result
	StateRes   []autofunc.Result
	OutputVecs []linalg.Vector
	StateVecs  []linalg.Vector
}

func (b *blockOutput) States() []linalg.Vector {
	return b.StateVecs
}

func (b *blockOutput) Outputs() []linalg.Vector {
	return b.OutputVecs
}

func (b *blockOutput) Gradient(u *rnn.UpstreamGradient, g autofunc.Gradient) {
	if u.Outputs != nil {
		for i, output := range b.OutputRes {
			output.PropagateGradient(u.Outputs[i], g)
		}
	}
	if u.States != nil {
		for i, state := range b.StateRes {
			state.PropagateGradient(u.States[i], g)
		}
	}
}