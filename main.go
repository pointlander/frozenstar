// Copyright 2024 The FrozenStar Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"

	"github.com/pointlander/frozenstar/kmeans"
	"github.com/pointlander/matrix"
)

const (
	// Size is the size of the workload
	Size = 40
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

func main() {
	rng := matrix.Rand(1)
	sets := Load()

	process := func(sample matrix.Sample) [][]float64 {
		x1 := sample.Vars[0][0].Sample()
		y1 := sample.Vars[0][1].Sample()
		z1 := sample.Vars[0][2].Sample()
		w1 := x1.Add(y1.H(z1))

		x2 := sample.Vars[1][0].Sample()
		y2 := sample.Vars[1][1].Sample()
		z2 := sample.Vars[1][2].Sample()
		b1 := x2.Add(y2.H(z2))

		x3 := sample.Vars[2][0].Sample()
		y3 := sample.Vars[2][1].Sample()
		z3 := sample.Vars[2][2].Sample()
		w2 := x3.Add(y3.H(z3))

		x4 := sample.Vars[3][0].Sample()
		y4 := sample.Vars[3][1].Sample()
		z4 := sample.Vars[3][2].Sample()
		b2 := x4.Add(y4.H(z4))

		rawData := make([][]float64, 0, 8)
		for _, set := range sets[:Size] {
			for _, t := range set.Train {
				output := matrix.NewMatrix(7, 1)
				for _, v := range t.Input {
					for _, s := range v {
						input := matrix.NewZeroMatrix(10, 1)
						input.Data[s] = 1
						in := matrix.NewMatrix(17, 1)
						in.Data = append(in.Data, input.Data...)
						in.Data = append(in.Data, output.Data...)
						output = w2.MulT(w1.MulT(in).Add(b1).Everett()).Add(b2)
					}
				}
				data := make([]float64, 0, 7)
				for _, value := range output.Data {
					data = append(data, float64(value))
				}
				rawData = append(rawData, data)
			}
		}

		meta := matrix.NewMatrix(len(rawData), len(rawData), make([]float32, len(rawData)*len(rawData))...)
		for i := 0; i < 100; i++ {
			clusters, _, err := kmeans.Kmeans(int64(i+1), rawData, len(sets), kmeans.SquaredEuclideanDistance, -1)
			if err != nil {
				panic(err)
			}
			for i := 0; i < len(rawData); i++ {
				target := clusters[i]
				for j, v := range clusters {
					if v == target {
						meta.Data[i*len(rawData)+j]++
					}
				}
			}
		}
		/*meta = matrix.SelfAttention(meta, meta, meta)*/

		x := make([][]float64, len(rawData))
		for i := range x {
			x[i] = make([]float64, len(rawData))
			for j := range x[i] {
				x[i][j] = float64(meta.Data[i*len(rawData)+j])
			}
		}

		return x
	}
	optimizer := matrix.NewOptimizer(&rng, 4, .1, 4, func(samples []matrix.Sample, x ...matrix.Matrix) {
		done := make(chan bool, 8)
		sample := func(s *matrix.Sample) {
			meta := process(*s)

			entropy := 0.0
			for i := range meta {
				sum := 0.0
				for _, value := range meta[i] {
					sum += value
				}
				if sum == 0 {
					continue
				}
				for _, value := range meta[i] {
					if value == 0 {
						continue
					}
					p := value / sum
					entropy += p * math.Log(p)
				}
			}
			s.Cost = -entropy / float64(len(meta))
			done <- true
		}
		index, flight, cpus := 0, 0, runtime.NumCPU()
		for flight < cpus && index < len(samples) {
			go sample(&samples[index])
			index++
			flight++
		}
		for index < len(samples) {
			<-done
			flight--
			fmt.Printf(".")

			go sample(&samples[index])
			index++
			flight++
		}
		for i := 0; i < flight; i++ {
			<-done
			fmt.Printf(".")
		}
		fmt.Printf("\n")
	}, matrix.NewCoord(17, 7), matrix.NewCoord(7, 1), matrix.NewCoord(14, 7), matrix.NewCoord(7, 1))
	var sample matrix.Sample
	for i := 0; i < 33; i++ {
		sample = optimizer.Iterate()
		fmt.Println(i, sample.Cost)
		cost := sample.Cost
		if cost < 0 {
			cost = 0
			break
		}
		if cost < 1e-9 {
			break
		}
	}

	meta := process(sample)
	clusters, _, err := kmeans.Kmeans(1, meta, len(sets), kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, v := range clusters {
		fmt.Printf("%3d %d\n", i, v)
	}
}
