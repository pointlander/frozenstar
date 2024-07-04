// Copyright 2024 The FrozenStar Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/pointlander/matrix"
)

// SA is self attention mode
func SA() {
	rng := matrix.Rand(1)

	sets := Load()
	get := func(s int) (offset, target int, opt []matrix.Matrix) {
		train, test := make([]Pair, 0, 8), make([]Pair, 0, 8)
		set := sets[s]
		for _, t := range set.Train {
			pair := Pair{
				Class: s,
			}
			for j, v := range t.Input {
				for i := range v {
					pair.Input = append(pair.Input, Pixel{
						C: v[i],
						X: i,
						Y: j,
					})
				}
			}
			for j, v := range t.Output {
				for i := range v {
					pair.Output = append(pair.Output, Pixel{
						C: v[i],
						X: i,
						Y: j,
					})
				}
			}
			train = append(train, pair)
		}
		for _, t := range set.Test {
			pair := Pair{
				Class: s,
			}
			for j, v := range t.Input {
				for i := range v {
					pair.Input = append(pair.Input, Pixel{
						C: v[i],
						X: i,
						Y: j,
					})
				}
			}
			for j, v := range t.Output {
				for i := range v {
					pair.Output = append(pair.Output, Pixel{
						C: v[i],
						X: i,
						Y: j,
					})
				}
			}
			test = append(test, pair)
		}
		opt = make([]matrix.Matrix, len(train))
		for i := range opt {
			offset = len(train[i].Input) + len(train[i].Output) + len(test[0].Input)
			target = len(test[0].Output)
			opt[i] = matrix.NewMatrix(Input, offset+target)
		}
		for i, pair := range train {
			for _, p := range pair.Input {
				input := matrix.NewZeroMatrix(Input, 1)
				input.Data[p.C] = 1
				input.Data[10+p.X] = 1
				input.Data[10+30+p.Y] = 1
				opt[i].Data = append(opt[i].Data, input.Data...)
			}
			for _, p := range pair.Output {
				input := matrix.NewZeroMatrix(Input, 1)
				input.Data[p.C] = 1
				input.Data[10+p.X] = 1
				input.Data[10+30+p.Y] = 1
				input.Data[10+30+30] = 1
				opt[i].Data = append(opt[i].Data, input.Data...)
			}

			for _, p := range test[0].Input {
				input := matrix.NewZeroMatrix(Input, 1)
				input.Data[p.C] = 1
				input.Data[10+p.X] = 1
				input.Data[10+30+p.Y] = 1
				opt[i].Data = append(opt[i].Data, input.Data...)
			}
			for _, p := range test[0].Output {
				input := matrix.NewZeroMatrix(Input, 1)
				_ = p
				/*input.Data[p.C] = 1
				input.Data[10+p.X] = 1
				input.Data[10+30+p.Y] = 1
				input.Data[10+30+30] = 1*/
				opt[i].Data = append(opt[i].Data, input.Data...)
			}
		}
		return offset, target, opt
	}
	offset, target, opt := get(0)
	optimizer := matrix.NewOptimizer(&rng, 8, .1, 1, func(samples []matrix.Sample, x ...matrix.Matrix) {
		for s, sample := range samples {
			x1 := sample.Vars[0][0].Sample()
			y1 := sample.Vars[0][1].Sample()
			z1 := sample.Vars[0][2].Sample()
			w1 := x1.Add(y1.H(z1))
			for j := range opt {
				for k, value := range w1.Data {
					bit := 1.0
					if value < 0.0 {
						bit = 0.0
					}
					opt[j].Data[Input*offset+k] = float32(bit)
				}
			}
			sum := 0.0
			for i := range opt {
				entropy := matrix.SelfEntropy64(opt[i], opt[i], opt[i])
				for _, e := range entropy {
					sum += e
				}
			}
			samples[s].Cost = sum
		}
	}, matrix.NewCoord(Input*target, 1))
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
}
