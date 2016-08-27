package statebrain

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/rnn"
)

func init() {
	var b Block
	serializer.RegisterTypedDeserializer(b.SerializerType(), DeserializeBlock)
}

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

// DeserializeBlock deserializes a Block.
func DeserializeBlock(d []byte) (*Block, error) {
	var res Block
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
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

// BatchR is like Batch but with r-operator support.
func (b *Block) BatchR(v autofunc.RVector, in *rnn.BlockRInput) rnn.BlockROutput {
	out := &blockROutput{
		OutputRes:   make([]autofunc.RResult, len(in.Inputs)),
		StateRes:    make([]autofunc.RResult, len(in.Inputs)),
		OutputVecs:  make([]linalg.Vector, len(in.Inputs)),
		StateVecs:   make([]linalg.Vector, len(in.Inputs)),
		OutputVecsR: make([]linalg.Vector, len(in.Inputs)),
		StateVecsR:  make([]linalg.Vector, len(in.Inputs)),
	}

	var softmax autofunc.Softmax
	for i, inputVec := range in.Inputs {
		inSoftmax := softmax.ApplyR(v, inputVec)
		var input int
		var maxValue float64
		for j, x := range inSoftmax.Output() {
			if x >= maxValue {
				maxValue = x
				input = j
			}
		}

		stateProbs := softmax.ApplyR(v, in.States[i])
		res := autofunc.PoolR(stateProbs, func(inState autofunc.RResult) autofunc.RResult {
			var output autofunc.RResult
			var newStates autofunc.RResult
			for stateIdx, entry := range b.Entries {
				prob := autofunc.SliceR(inState, stateIdx, stateIdx+1)
				outVar := autofunc.NewRVariable(entry.Output, v)
				scaledOut := autofunc.ScaleFirstR(outVar, prob)
				transVar := autofunc.NewRVariable(entry.Transitions[input], v)
				scaledStates := autofunc.ScaleFirstR(transVar, prob)
				if output == nil {
					output = scaledOut
					newStates = scaledStates
				} else {
					output = autofunc.AddR(output, scaledOut)
					newStates = autofunc.AddR(newStates, scaledOut)
				}
			}
			return autofunc.ConcatR(output, newStates)
		})
		out.OutputRes[i] = autofunc.SliceR(res, 0, len(inputVec.Output()))
		out.OutputVecs[i] = out.OutputRes[i].Output()
		out.OutputVecsR[i] = out.OutputRes[i].ROutput()
		out.StateRes[i] = autofunc.SliceR(res, len(inputVec.Output()), len(res.Output()))
		out.StateVecs[i] = out.StateRes[i].Output()
		out.StateVecsR[i] = out.StateRes[i].ROutput()
	}

	return out
}

// Parameters returns all of the variables involved in
// this model.
func (b *Block) Parameters() []*autofunc.Variable {
	var res []*autofunc.Variable
	for _, e := range b.Entries {
		res = append(res, e.Output)
		res = append(res, e.Transitions...)
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// this block with the serializer package.
func (b *Block) SerializerType() string {
	return "github.com/unixpickle/statebrain.Block"
}

// Serialize serializes this block.
func (b *Block) Serialize() ([]byte, error) {
	return json.Marshal(b)
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

type blockROutput struct {
	OutputRes   []autofunc.RResult
	StateRes    []autofunc.RResult
	OutputVecs  []linalg.Vector
	StateVecs   []linalg.Vector
	OutputVecsR []linalg.Vector
	StateVecsR  []linalg.Vector
}

func (b *blockROutput) States() []linalg.Vector {
	return b.StateVecs
}

func (b *blockROutput) RStates() []linalg.Vector {
	return b.StateVecsR
}

func (b *blockROutput) Outputs() []linalg.Vector {
	return b.OutputVecs
}

func (b *blockROutput) ROutputs() []linalg.Vector {
	return b.OutputVecsR
}

func (b *blockROutput) RGradient(u *rnn.UpstreamRGradient, rg autofunc.RGradient,
	g autofunc.Gradient) {
	if u.Outputs != nil {
		for i, output := range b.OutputRes {
			output.PropagateRGradient(u.Outputs[i], u.ROutputs[i], rg, g)
		}
	}
	if u.States != nil {
		for i, state := range b.StateRes {
			state.PropagateRGradient(u.States[i], u.RStates[i], rg, g)
		}
	}
}
