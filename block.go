package statebrain

import (
	"encoding/json"
	"math/rand"

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

type blockState struct {
	State autofunc.Result
}

type blockRState struct {
	RState autofunc.RResult
}

// A StateEntry represents one fuzzy Markov state in
// a bigger statebrain model.
type StateEntry struct {
	Output      *autofunc.Variable
	Transitions []*autofunc.Variable
}

// A Block is an rnn.Block for the statebrain model.
type Block struct {
	StartVar *autofunc.Variable
	Entries  []StateEntry
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
	res := &Block{
		Entries: make([]StateEntry, stateCount),
		StartVar: &autofunc.Variable{
			Vector: make(linalg.Vector, stateCount),
		},
	}

	// Make the first state much more likely.
	res.StartVar.Vector[0] = 100

	for i := range res.Entries {
		res.Entries[i].Output = &autofunc.Variable{
			Vector: make(linalg.Vector, alphabetSize),
		}
		for j := range res.Entries[i].Output.Vector {
			res.Entries[i].Output.Vector[j] = rand.NormFloat64()
		}
		res.Entries[i].Transitions = make([]*autofunc.Variable, alphabetSize)
		for j := range res.Entries[i].Transitions {
			res.Entries[i].Transitions[j] = &autofunc.Variable{
				Vector: make(linalg.Vector, stateCount),
			}
			for k := range res.Entries[i].Transitions[j].Vector {
				res.Entries[i].Transitions[j].Vector[k] = rand.NormFloat64() * 2
			}
		}
	}
	return res
}

// StartState returns the initial state distribution.
func (b *Block) StartState() rnn.State {
	softmax := neuralnet.LogSoftmaxLayer{}
	return blockState{State: softmax.Apply(b.StartVar)}
}

// StartStateR returns the initial state.
func (b *Block) StartRState(rv autofunc.RVector) rnn.RState {
	softmax := neuralnet.LogSoftmaxLayer{}
	res := softmax.ApplyR(rv, autofunc.NewRVariable(b.StartVar, rv))
	return blockRState{
		RState: res,
	}
}

// PropagateStart performs back-propagation through the
// start state.
func (b *Block) PropagateStart(s []rnn.State, u []rnn.StateGrad, grad autofunc.Gradient) {
	for i, g := range u {
		res := s[i].(blockState).State
		vec := g.(rnn.VecStateGrad)
		res.PropagateGradient(linalg.Vector(vec), grad)
	}
}

// PropagateStartR is like PropagateStart, but for an
// RStateGrad.
// The g argument may be nil.
func (b *Block) PropagateStartR(s []rnn.RState, u []rnn.RStateGrad, rg autofunc.RGradient,
	g autofunc.Gradient) {
	for i, sgObj := range u {
		res := s[i].(blockRState).RState
		sg := sgObj.(rnn.VecRStateGrad)
		res.PropagateRGradient(sg.State, sg.RState, rg, g)
	}
}

// ApplyBlock applies the block to a batch of inputs.
func (b *Block) ApplyBlock(s []rnn.State, in []autofunc.Result) rnn.BlockResult {
	out := &blockResult{
		Pool:       make([]*autofunc.Variable, len(in)),
		OutputRes:  make([]autofunc.Result, len(in)),
		StateRes:   make([]autofunc.Result, len(in)),
		OutputVecs: make([]linalg.Vector, len(in)),
		StatesOut:  make([]rnn.State, len(in)),
	}

	var softmax neuralnet.LogSoftmaxLayer
	for i, rawState := range s {
		out.Pool[i] = &autofunc.Variable{
			Vector: rawState.(blockState).State.Output(),
		}
		state := out.Pool[i]
		input := maxIndex(in[i].Output())
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
		out.StatesOut[i] = blockState{State: out.StateRes[i]}
	}

	return out
}

// ApplyBlockR is like ApplyBlock, but with support for
// the R operator.
func (b *Block) ApplyBlockR(v autofunc.RVector, s []rnn.RState,
	in []autofunc.RResult) rnn.BlockRResult {
	out := &blockRResult{
		Pool:        make([]*autofunc.Variable, len(s)),
		OutputRes:   make([]autofunc.RResult, len(in)),
		StateRes:    make([]autofunc.RResult, len(in)),
		OutputVecs:  make([]linalg.Vector, len(in)),
		OutputVecsR: make([]linalg.Vector, len(in)),
		StatesOut:   make([]rnn.RState, len(in)),
	}

	var softmax neuralnet.LogSoftmaxLayer
	for i, rawState := range s {
		out.Pool[i] = &autofunc.Variable{
			Vector: rawState.(blockRState).RState.Output(),
		}
		state := &autofunc.RVariable{
			Variable:   out.Pool[i],
			ROutputVec: rawState.(blockRState).RState.ROutput(),
		}
		input := maxIndex(in[i].Output())
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
		out.StatesOut[i] = blockRState{RState: out.StateRes[i]}
	}

	return out
}

// Parameters returns all of the variables involved in
// this model.
func (b *Block) Parameters() []*autofunc.Variable {
	res := []*autofunc.Variable{b.StartVar}
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

type blockResult struct {
	Pool       []*autofunc.Variable
	OutputRes  []autofunc.Result
	StateRes   []autofunc.Result
	OutputVecs []linalg.Vector
	StatesOut  []rnn.State
}

func (b *blockResult) States() []rnn.State {
	return b.StatesOut
}

func (b *blockResult) Outputs() []linalg.Vector {
	return b.OutputVecs
}

func (b *blockResult) PropagateGradient(u []linalg.Vector, s []rnn.StateGrad,
	g autofunc.Gradient) []rnn.StateGrad {
	return rnn.PropagateVecStatePool(g, b.Pool, func() {
		if u != nil {
			for i, output := range b.OutputRes {
				output.PropagateGradient(copyVec(u[i]), g)
			}
		}
		if s != nil {
			for i, state := range b.StateRes {
				if s[i] != nil {
					ups := linalg.Vector(s[i].(rnn.VecStateGrad))
					state.PropagateGradient(copyVec(ups), g)
				}
			}
		}
	})
}

type blockRResult struct {
	Pool        []*autofunc.Variable
	OutputRes   []autofunc.RResult
	StateRes    []autofunc.RResult
	OutputVecs  []linalg.Vector
	OutputVecsR []linalg.Vector
	StatesOut   []rnn.RState
}

func (b *blockRResult) RStates() []rnn.RState {
	return b.StatesOut
}

func (b *blockRResult) Outputs() []linalg.Vector {
	return b.OutputVecs
}

func (b *blockRResult) ROutputs() []linalg.Vector {
	return b.OutputVecsR
}

func (b *blockRResult) PropagateRGradient(u, uR []linalg.Vector, s []rnn.RStateGrad,
	rg autofunc.RGradient, g autofunc.Gradient) []rnn.RStateGrad {
	return rnn.PropagateVecRStatePool(rg, g, b.Pool, func() {
		if u != nil {
			for i, output := range b.OutputRes {
				output.PropagateRGradient(copyVec(u[i]), copyVec(uR[i]), rg, g)
			}
		}
		if s != nil {
			for i, state := range b.StateRes {
				if s[i] != nil {
					ups := s[i].(rnn.VecRStateGrad)
					state.PropagateRGradient(copyVec(ups.State),
						copyVec(ups.RState), rg, g)
				}
			}
		}
	})
}

func copyVec(x linalg.Vector) linalg.Vector {
	return append(linalg.Vector{}, x...)
}
