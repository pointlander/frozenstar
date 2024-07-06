// Copyright 2024 The FrozenStar Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"

	"github.com/pointlander/matrix"
)

// Opt is an optimization
type Opt struct {
	Opt    matrix.Matrix
	Input  Pair
	Output Pair
}

// TargetOffset is the target offset
func (o Opt) TargetOffset() int {
	return len(o.Input.Input.I) + len(o.Input.Output.I) + len(o.Output.Input.I)
}

// TargetSize is the size of the target
func (o Opt) TargetSize() int {
	return len(o.Output.Output.I)
}

// SA is self attention mode
func SA() {
	rng := matrix.Rand(1)

	sets := Load()
	get := func(s int) (opt []Opt) {
		train, test := make([]Pair, 0, 8), make([]Pair, 0, 8)
		set := sets[s]
		for _, t := range set.Train {
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
					pair.Input.I = append(pair.Input.I, Pixel{
						C: v[i],
						X: i,
						Y: j,
					})
				}
			}
			for j, v := range t.Output {
				for i := range v {
					pair.Output.I = append(pair.Output.I, Pixel{
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
					pair.Input.I = append(pair.Input.I, Pixel{
						C: v[i],
						X: i,
						Y: j,
					})
				}
			}
			for j, v := range t.Output {
				for i := range v {
					pair.Output.I = append(pair.Output.I, Pixel{
						C: v[i],
						X: i,
						Y: j,
					})
				}
			}
			test = append(test, pair)
		}
		opt = make([]Opt, len(train))
		for i := range opt {
			opt[i].Input = train[i]
			opt[i].Output = test[0]
			opt[i].Opt = matrix.NewZeroMatrix(Input, opt[i].TargetOffset()+opt[i].TargetSize())
		}
		for i, pair := range train {
			for _, p := range pair.Input.I {
				opt[i].Opt.Data[p.C] = 1
				opt[i].Opt.Data[10+p.X] = 1
				opt[i].Opt.Data[10+30+p.Y] = 1
			}
			for _, p := range pair.Output.I {
				opt[i].Opt.Data[Input+p.C] = 1
				opt[i].Opt.Data[Input+10+p.X] = 1
				opt[i].Opt.Data[Input+10+30+p.Y] = 1
				opt[i].Opt.Data[Input+10+30+30] = 1
			}

			for _, p := range test[0].Input.I {
				opt[i].Opt.Data[2*Input+p.C] = 1
				opt[i].Opt.Data[2*Input+10+p.X] = 1
				opt[i].Opt.Data[2*Input+10+30+p.Y] = 1
			}
		}
		return opt
	}

	done := make(chan bool, 8)
	process := func(sample *matrix.Sample) {
		opt := get(0)
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
			offset := len(opt[j].Input.Input.I) + len(opt[j].Input.Output.I) + len(opt[j].Output.Input.I)
			for k, value := range w1.Data {
				bit := 1.0
				if value < 0.0 {
					bit = 0.0
				}
				opt[j].Opt.Data[Input*offset+k] = float32(bit)
			}
		}
		sum := 0.0
		for i := range opt {
			entropy := matrix.SelfEntropy64(q.MulT(opt[i].Opt), k.MulT(opt[i].Opt), v.MulT(opt[i].Opt))
			for _, e := range entropy {
				sum += e
			}
		}
		sample.Cost = sum
		done <- true
	}
	opt := get(0)
	optimizer := matrix.NewOptimizer(&rng, 8, .1, 4, func(samples []matrix.Sample, x ...matrix.Matrix) {
		index, flight, cpus := 0, 0, runtime.NumCPU()
		for flight < cpus && index < len(samples) {
			go process(&samples[index])
			index++
			flight++
		}
		for index < len(samples) {
			<-done
			flight--
			fmt.Printf(".")

			go process(&samples[index])
			index++
			flight++
		}
		for i := 0; i < flight; i++ {
			<-done
			fmt.Printf(".")
		}
		fmt.Printf("\n")
	}, matrix.NewCoord(Input*opt[0].TargetSize(), 1), matrix.NewCoord(Input, Input), matrix.NewCoord(Input, Input), matrix.NewCoord(Input, Input))
	w, h := opt[0].Output.Output.W, opt[0].Output.Output.H
	var sample matrix.Sample
	for i := 0; i < 33; i++ {
		sample = optimizer.Iterate()
		fmt.Println(i, sample.Cost)
		cost := sample.Cost
		type Result struct {
			Color  byte
			Signal float32
		}
		grid := make([][]Result, h)
		for j := range grid {
			grid[j] = make([]Result, w)
		}
		x1 := sample.Vars[0][0].Sample()
		y1 := sample.Vars[0][1].Sample()
		z1 := sample.Vars[0][2].Sample()
		w1 := x1.Add(y1.H(z1))
		for offset := 0; offset < len(w1.Data); offset += Input {
			maxColor, color := float32(0.0), 0
			cc := w1.Data[offset : offset+10]
			for j := range cc {
				for cc[j] > maxColor {
					maxColor, color = cc[j], j
				}
			}
			xx := w1.Data[offset+10 : offset+10+w]
			maxX := float32(0.0)
			x := 0
			for j := range xx {
				for xx[j] > maxX {
					maxX, x = xx[j], j
				}
			}
			yy := w1.Data[offset+10+w : offset+10+w+h]
			maxY := float32(0.0)
			y := 0
			for j := range yy {
				for yy[j] > maxY {
					maxY, y = yy[j], j
				}
			}
			if maxColor > grid[y][x].Signal {
				grid[y][x].Signal = maxColor
				grid[y][x].Color = byte(color)
			}
		}
		index := 0
		sum, total := 0.0, 0.0
		for _, v := range grid {
			for _, value := range v {
				color := opt[0].Output.Output.I[index].C
				if color == value.Color {
					sum++
					fmt.Printf("* ")
				} else {
					fmt.Printf("%d ", value.Color)
				}
				index++
				total++
			}
			fmt.Println()
		}
		fmt.Println("accuracy", sum/total)
		if cost < 0 {
			cost = 0
			break
		}
		if cost < 1e-9 {
			break
		}
	}
}
