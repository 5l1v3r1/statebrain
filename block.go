package statebrain

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
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

	var softmax neuralnet.LogSoftmaxLayer
	for i, state := range in.States {
		input := maxIndex(in.Inputs[i].Output())

		// The initial state is always the zero vector.
		if allZeroes(state.Output()) {
			state = b.initialState()
		}

		var output autofunc.Result
		var newStates autofunc.Result
		for stateIdx, entry := range b.Entries {
			outputs := softmax.Apply(entry.Output)
			transitions := softmax.Apply(entry.Transitions[input])

			probLog := autofunc.Slice(state, stateIdx, stateIdx+1)
			scaledOut := autofunc.AddFirst(outputs, probLog)
			scaledStates := autofunc.AddFirst(transitions, probLog)

			if output == nil {
				output = scaledOut
				newStates = scaledStates
			} else {
				output = autofunc.AddLogDomain(output, scaledOut)
				newStates = autofunc.AddLogDomain(newStates, scaledStates)
			}
		}

		out.OutputRes[i] = output
		out.OutputVecs[i] = out.OutputRes[i].Output()
		out.StateRes[i] = newStates
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

	var softmax neuralnet.LogSoftmaxLayer
	for i, state := range in.States {
		input := maxIndex(in.Inputs[i].Output())

		// The initial state is always the zero vector.
		if allZeroes(state.Output()) {
			state = b.initialStateR()
		}

		var output autofunc.RResult
		var newStates autofunc.RResult
		for stateIdx, entry := range b.Entries {
			outputs := softmax.ApplyR(v, autofunc.NewRVariable(entry.Output, v))
			transitions := softmax.ApplyR(v, autofunc.NewRVariable(entry.Transitions[input], v))

			probLog := autofunc.SliceR(state, stateIdx, stateIdx+1)
			scaledOut := autofunc.AddFirstR(outputs, probLog)
			scaledStates := autofunc.AddFirstR(transitions, probLog)

			if output == nil {
				output = scaledOut
				newStates = scaledStates
			} else {
				output = autofunc.AddLogDomainR(output, scaledOut)
				newStates = autofunc.AddLogDomainR(newStates, scaledStates)
			}
		}

		out.OutputRes[i] = output
		out.OutputVecs[i] = out.OutputRes[i].Output()
		out.OutputVecsR[i] = out.OutputRes[i].ROutput()
		out.StateRes[i] = newStates
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

// initialState returns the initial state variable.
func (b *Block) initialState() *autofunc.Variable {
	// Start at the first state with almost 100% chance.
	// Perfect accuracy (100% probability) would lead to
	// -Infinity values in the log probability vector.
	nonLogProbs := make(linalg.Vector, len(b.Entries))
	nonLogProbs[0] = 100
	softmax := neuralnet.LogSoftmaxLayer{}
	return &autofunc.Variable{
		Vector: softmax.Apply(&autofunc.Variable{Vector: nonLogProbs}).Output(),
	}
}

func (b *Block) initialStateR() *autofunc.RVariable {
	return autofunc.NewRVariable(b.initialState(), autofunc.RVector{})
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
