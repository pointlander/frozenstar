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
	get := func(s int) (w, h, offset, target int, opt []matrix.Matrix) {
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
			h = len(set.Test[0].Output)
			w = len(set.Test[0].Output[0])
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
		return w, h, offset, target, opt
	}
	w, h, offset, target, opt := get(0)
	optimizer := matrix.NewOptimizer(&rng, 8, .1, 4, func(samples []matrix.Sample, x ...matrix.Matrix) {
		for s, sample := range samples {
			x1 := sample.Vars[0][0].Sample()
			y1 := sample.Vars[0][1].Sample()
			z1 := sample.Vars[0][2].Sample()
			w1 := x1.Add(y1.H(z1))
			x2 := sample.Vars[1][0].Sample()
			y2 := sample.Vars[1][1].Sample()
			z2 := sample.Vars[1][2].Sample()
			q := x2.Add(y2.H(z2))
			x3 := sample.Vars[2][0].Sample()
			y3 := sample.Vars[2][1].Sample()
			z3 := sample.Vars[2][2].Sample()
			k := x3.Add(y3.H(z3))
			x4 := sample.Vars[3][0].Sample()
			y4 := sample.Vars[3][1].Sample()
			z4 := sample.Vars[3][2].Sample()
			v := x4.Add(y4.H(z4))
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
				entropy := matrix.SelfEntropy64(q.MulT(opt[i]), k.MulT(opt[i]), v.MulT(opt[i]))
				for _, e := range entropy {
					sum += e
				}
			}
			samples[s].Cost = sum
		}
	}, matrix.NewCoord(Input*target, 1), matrix.NewCoord(Input, Input), matrix.NewCoord(Input, Input), matrix.NewCoord(Input, Input))
	var sample matrix.Sample
	for i := 0; i < 33; i++ {
		sample = optimizer.Iterate()
		fmt.Println(i, sample.Cost)
		cost := sample.Cost
		grid := make([][]byte, h)
		for j := range grid {
			grid[j] = make([]byte, w)
		}
		x1 := sample.Vars[0][0].Sample()
		y1 := sample.Vars[0][1].Sample()
		z1 := sample.Vars[0][2].Sample()
		w1 := x1.Add(y1.H(z1))
		for offset := 0; offset < len(w1.Data); offset += Input {
			max, color := float32(0.0), 0
			cc := w1.Data[offset : offset+10]
			for j := range cc {
				for cc[j] > max {
					max, color = cc[j], j
				}
			}
			xx := w1.Data[offset+10 : offset+10+w]
			max = 0
			x := 0
			for j := range xx {
				for xx[j] > max {
					max, x = xx[j], j
				}
			}
			yy := w1.Data[offset+10+w : offset+10+w+h]
			max = 0
			y := 0
			for j := range yy {
				for yy[j] > max {
					max, y = yy[j], j
				}
			}
			grid[y][x] = byte(color)
		}
		for _, v := range grid {
			for _, value := range v {
				fmt.Printf("%d ", value)
			}
			fmt.Println()
		}
		if cost < 0 {
			cost = 0
			break
		}
		if cost < 1e-9 {
			break
		}
	}
}
