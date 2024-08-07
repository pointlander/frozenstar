// Copyright 2024 The FrozenStar Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"runtime"

	"github.com/pointlander/frozenstar/kmeans"
	"github.com/pointlander/matrix"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// Encdec encoder decoder model
func Encdec() {
	rng := matrix.Rand(1)
	sets := Load()

	pairs := make([]Pair, 0, 8)
	for s, set := range sets[:Size] {
		for _, t := range set.Train {
			direction := false
			pair := Pair{
				Class: s,
				Input: Image{
					W: len(t.Input[0]),
					H: len(t.Input),
				},
				Output: Image{
					W: len(t.Output[0]),
					H: len(t.Output),
				},
			}
			for j, v := range t.Input {
				for i := range v {
					if direction {
						i = len(v) - i - 1
					}
					pair.Input.I = append(pair.Input.I, Pixel{
						C: v[i],
						X: i,
						Y: j,
					})
				}
				direction = !direction
			}
			direction = false
			for j, v := range t.Output {
				for i := range v {
					if direction {
						i = len(v) - i - 1
					}
					pair.Output.I = append(pair.Output.I, Pixel{
						C: v[i],
						X: i,
						Y: j,
					})
				}
				direction = !direction
			}
			pairs = append(pairs, pair)
		}
	}

	process := func(sample matrix.Sample) ([][]float64, []int, []matrix.Matrix) {
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
		params := []matrix.Matrix{w1, b1, w2, b2}

		rawData := make([][]float64, 0, 8)
		classes := make([]int, 0, 8)
		for _, pair := range pairs {
			output := matrix.NewZeroMatrix(Output, 1)
			for _, p := range pair.Input.I {
				input := matrix.NewZeroMatrix(Input, 1)
				input.Data[p.C] = 1
				input.Data[10+p.X] = 1
				input.Data[10+30+p.Y] = 1
				in := matrix.NewMatrix(Input+Output, 1)
				in.Data = append(in.Data, input.Data...)
				in.Data = append(in.Data, output.Data...)
				output = w2.MulT(w1.MulT(in).Add(b1).Everett()).Add(b2).Sigmoid()
			}
			data := make([]float64, 0, 7)
			for _, value := range output.Data {
				data = append(data, float64(value))
			}
			rawData = append(rawData, data)
			classes = append(classes, pair.Class)
		}

		meta := matrix.NewMatrix(len(rawData), len(rawData), make([]float32, len(rawData)*len(rawData))...)
		for i := 0; i < 100; i++ {
			clusters, _, err := kmeans.Kmeans(int64(i+1), rawData, len(sets[:Size]), kmeans.SquaredEuclideanDistance, -1)
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
		meta = matrix.SelfAttention(meta, meta, meta)

		x := make([][]float64, len(rawData))
		for i := range x {
			x[i] = make([]float64, len(rawData))
			for j := range x[i] {
				x[i][j] = float64(meta.Data[i*len(rawData)+j])
			}
		}

		return x, classes, params
	}
	optimizer := matrix.NewOptimizer(&rng, 4, .1, 4, func(samples []matrix.Sample, x ...matrix.Matrix) {
		done := make(chan bool, 8)
		sample := func(s *matrix.Sample) {
			meta, _, _ := process(*s)

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
	}, matrix.NewCoord(Input+Output, Width), matrix.NewCoord(Width, 1),
		matrix.NewCoord(2*Width, Output), matrix.NewCoord(Output, 1))
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

	clustersCount := len(sets[:Size])
	meta, classes, params := process(sample)
	clusters, _, err := kmeans.Kmeans(1, meta, clustersCount, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, v := range clusters {
		fmt.Printf("%3d %3d %d\n", i, classes[i], v)
	}

	var values plotter.Values
	for _, param := range params {
		for _, value := range param.Data {
			values = append(values, float64(value))
		}
	}
	p := plot.New()
	p.Title.Text = "histogram plot"

	hist, err := plotter.NewHist(values, 20)
	if err != nil {
		panic(err)
	}
	p.Add(hist)

	if err := p.Save(8*vg.Inch, 8*vg.Inch, "histogram.png"); err != nil {
		panic(err)
	}

	ab, ba := make([][]float64, clustersCount), make([][]float64, clustersCount)
	for i := range ab {
		ab[i] = make([]float64, clustersCount)
		ba[i] = make([]float64, clustersCount)
	}
	for i := range clusters {
		a := int(classes[i])
		b := clusters[i]
		ab[a][b]++
		ba[b][a]++
	}
	entropy := 0.0
	for i := 0; i < clustersCount; i++ {
		entropy += (1.0 / float64(clustersCount)) * math.Log(1.0/float64(clustersCount))
	}
	fmt.Println(-entropy, -(1.0/float64(clustersCount))*math.Log(1.0/float64(clustersCount)))
	sumAB := 0.0
	for i := range ab {
		entropy := 0.0
		for _, value := range ab[i] {
			if value > 0 {
				p := value / 150
				entropy += p * math.Log(p)
			}
		}
		entropy = -entropy
		fmt.Println("ab", i, entropy)
		sumAB += entropy
	}
	sumBA := 0.0
	for i := range ba {
		entropy := 0.0
		for _, value := range ba[i] {
			if value > 0 {
				p := value / 150
				entropy += p * math.Log(p)
			}
		}
		entropy = -entropy
		fmt.Println("ba", i, entropy)
		sumBA += entropy
	}
	fmt.Println("sumAB", sumAB)
	fmt.Println("sumBA", sumBA)

	outputs := []matrix.Matrix{}
	for _, pair := range pairs {
		output := matrix.NewZeroMatrix(Output, 1)
		for _, p := range pair.Input.I {
			input := matrix.NewZeroMatrix(Input, 1)
			input.Data[p.C] = 1
			input.Data[10+p.X] = 1
			input.Data[10+30+p.Y] = 1
			in := matrix.NewMatrix(Input+Output, 1)
			in.Data = append(in.Data, input.Data...)
			in.Data = append(in.Data, output.Data...)
			output = params[2].MulT(params[0].MulT(in).Add(params[1]).Everett()).Add(params[3]).Sigmoid()
		}
		data := matrix.NewMatrix(Output, 1)
		data.Data = append(data.Data, output.Data...)
		outputs = append(outputs, data)
	}
	processDecoder := func(sample matrix.Sample) (float64, []matrix.Matrix) {
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
		params := []matrix.Matrix{w1, b1, w2, b2}

		cost := 0.0
		for k, pair := range pairs {
			output := matrix.NewZeroMatrix(Input+Output, 1)
			copy(output.Data[Input:], outputs[k].Data)
			loss, count := 0.0, 0.0
			for i, p := range pair.Output.I {
				input := matrix.NewZeroMatrix(Input, 1)
				input.Data[p.C] = 1
				input.Data[10+p.X] = 1
				input.Data[10+30+p.Y] = 1
				in := matrix.NewMatrix(Output, 1)
				in.Data = append(in.Data, output.Data[Input:]...)
				if i > 0 {
					in = in.Sigmoid()
				}
				output = w2.MulT(w1.MulT(in).Add(b1).Everett()).Add(b2)
				for k := range output.Data[:Input] {
					diff := float64(input.Data[k]) - float64(output.Data[k])
					loss += diff * diff
					count++
				}
			}
			cost += loss / count
		}
		cost /= float64(Size)
		return cost, params
	}
	optimizerDecoder := matrix.NewOptimizer(&rng, 4, .1, 4, func(samples []matrix.Sample, x ...matrix.Matrix) {
		done := make(chan bool, 8)
		sample := func(s *matrix.Sample) {
			loss, _ := processDecoder(*s)
			s.Cost = loss
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
	}, matrix.NewCoord(Output, Width), matrix.NewCoord(Width, 1),
		matrix.NewCoord(2*Width, Input+Output), matrix.NewCoord(Input+Output, 1))
	var sample1 matrix.Sample
	for i := 0; i < 128; i++ {
		sample1 = optimizerDecoder.Iterate()
		fmt.Println(i, sample1.Cost)
		cost := sample.Cost
		if cost < 0 {
			cost = 0
			break
		}
		if cost < 1e-9 {
			break
		}
	}
	_ = sample1
}
