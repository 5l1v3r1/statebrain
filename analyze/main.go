package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"sort"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/statebrain"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "block_file")
		os.Exit(1)
	}
	data, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read file:", err)
		os.Exit(1)
	}
	block, err := statebrain.DeserializeBlock(data)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to deserialize:", err)
		os.Exit(1)
	}

	fmt.Print("Start states:")
	printProbIndices(block.StartVar)

	for stateIdx, entry := range block.Entries {
		fmt.Printf("State %d outputs:", stateIdx)
		printProbIndices(entry.Output)
	}
}

func printProbIndices(probVector *autofunc.Variable) {
	probs := newProbabilitySorter(probVector)
	sort.Sort(probs)
	var probSum float64
	for i, prob := range probs.probabilities {
		probSum += prob
		fmt.Printf(" %d (0x%x) (p=%.04f)", probs.indices[i], probs.indices[i], prob)
		if probSum > 0.95 {
			break
		}
	}
	fmt.Println()
}

type probabilitySorter struct {
	probabilities linalg.Vector
	indices       []int
}

func newProbabilitySorter(preSoftmax *autofunc.Variable) *probabilitySorter {
	softmax := autofunc.Softmax{}
	output := softmax.Apply(preSoftmax).Output()
	res := &probabilitySorter{
		probabilities: output,
		indices:       make([]int, len(output)),
	}
	for i := range output {
		res.indices[i] = i
	}
	return res
}

func (p *probabilitySorter) Len() int {
	return len(p.probabilities)
}

func (p *probabilitySorter) Swap(i, j int) {
	p.probabilities[i], p.probabilities[j] = p.probabilities[j], p.probabilities[i]
	p.indices[i], p.indices[j] = p.indices[j], p.indices[i]
}

func (p *probabilitySorter) Less(i, j int) bool {
	return p.probabilities[i] > p.probabilities[j]
}
