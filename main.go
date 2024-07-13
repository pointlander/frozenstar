// Copyright 2024 The FrozenStar Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
)

const (
	// Size is the size of the workload
	Size = 40
	// Input is the size of the input
	Input = 10 + 30 + 30 + 1
	// Width is the width of the network
	Width = 16
	// Output is the size of the output
	Output = 7
)

// Example is a learning example
type Example struct {
	Input  [][]byte `json:"input"`
	Output [][]byte `json:"output"`
}

// Set is a set of examples
type Set struct {
	Test  []Example `json:"test"`
	Train []Example `json:"train"`
}

// Load loads the data
func Load() []Set {
	dirs, err := os.ReadDir("ARC-AGI/data/training/")
	if err != nil {
		panic(err)
	}
	sets := make([]Set, len(dirs))
	for i, dir := range dirs {
		data, err := os.ReadFile("ARC-AGI/data/training/" + dir.Name())
		if err != nil {
			panic(err)
		}
		err = json.Unmarshal(data, &sets[i])
		if err != nil {
			panic(err)
		}
	}
	fmt.Println("loaded", len(sets))
	test, train := 0, 0
	for _, set := range sets {
		test += len(set.Test)
		train += len(set.Train)
	}
	fmt.Println("test", test)
	fmt.Println("train", train)
	return sets
}

// Pixel is an image pixel
type Pixel struct {
	C uint8
	X int
	Y int
}

// Image is an image
type Image struct {
	W int
	H int
	I []Pixel
}

// Pair is an input output pair
type Pair struct {
	Class  int
	Input  Image
	Output Image
}

var (
	// FlagCluster clustering mode
	FlagCluster = flag.Bool("cluster", false, "clustering mode")
	// FlagEncdec encoder decoder model
	FlagEncdec = flag.Bool("encdec", false, "encoder decoder model")
	// FlagSA self attention model
	FlagSA = flag.Bool("sa", false, "self attention model")
	// FlagAC is an autocoder model
	FlagAC = flag.Bool("ac", false, "autocoder model")
	// FlagSets is the number of sets to learn with
	FlagSets = flag.Int("sets", 2, "number of sets to learn with")
)

func main() {
	flag.Parse()

	if *FlagCluster {
		Cluster()
		return
	} else if *FlagEncdec {
		Encdec()
		return
	} else if *FlagSA {
		SA()
		return
	} else if *FlagAC {
		AC()
		return
	}
}
