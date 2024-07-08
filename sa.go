// Copyright 2024 The FrozenStar Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
	"sort"

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

// GetTrainingData gets the training data
func GetTrainingData(sets []Set, s int) (opt []Opt) {
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

// SA is self attention mode
func SA() {
	rng := matrix.Rand(1)

	sets := Load()
	done := make(chan bool, 8)
	process := func(sample *matrix.Sample) {
		opt := GetTrainingData(sets, 0)
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
		x5 := sample.Vars[4][0].Sample()
		y5 := sample.Vars[4][1].Sample()
		z5 := sample.Vars[4][2].Sample()
		w2 := x5.Add(y5.H(z5))
		x6 := sample.Vars[5][0].Sample()
		y6 := sample.Vars[5][1].Sample()
		z6 := sample.Vars[5][2].Sample()
		b2 := x6.Add(y6.H(z6))
		w, h := opt[0].Output.Output.W, opt[0].Output.Output.H
		for j := range opt {
			offset := len(opt[j].Input.Input.I) + len(opt[j].Input.Output.I) + len(opt[j].Output.Input.I)
			cc := opt[j].Opt.Data[Input*offset : Input*offset+10]
			w1CC := w1.Data[:10]
			maxCC, indexCC := float32(0.0), 0
			for key, value := range w1CC {
				if value > maxCC {
					indexCC, maxCC = key, value
				}
			}
			xx := opt[j].Opt.Data[Input*offset+10 : Input*offset+10+w]
			w1XX := w1.Data[10 : 10+w]
			maxXX, indexXX := float32(0.0), 0
			for key, value := range w1XX {
				if value > maxXX {
					indexXX, maxXX = key, value
				}
			}
			yy := opt[j].Opt.Data[Input*offset+10+w : Input*offset+10+w+h]
			w1YY := w1.Data[10+w : 10+w+h]
			maxYY, indexYY := float32(0.0), 0
			for key, value := range w1YY {
				if value > maxYY {
					indexYY, maxYY = key, value
				}
			}
			cc[indexCC] = 1
			xx[indexXX] = 1
			yy[indexYY] = 1
			opt[j].Opt.Data[Input*offset+10+w+h] = 1
			/*for k, value := range w1.Data {
				bit := 1.0
				if value < 0.0 {
					bit = 0.0
				}
				opt[j].Opt.Data[Input*offset+k] = float32(bit)
			}*/
		}
		sum := 0.0
		for i := range opt {
			output := w2.MulT(opt[i].Opt).Add(b2).Sigmoid()
			entropy := matrix.SelfEntropy64(q.MulT(output), k.MulT(output), v.MulT(output))
			for _, e := range entropy {
				sum += e
			}
		}
		sample.Cost = sum
		done <- true
	}
	opt := GetTrainingData(sets, 0)
	optimizer := matrix.NewOptimizer(&rng, 9, .1, 6, func(samples []matrix.Sample, x ...matrix.Matrix) {
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
	}, matrix.NewCoord(Input*opt[0].TargetSize(), 1),
		matrix.NewCoord(Input, 2*Input), matrix.NewCoord(Input, 2*Input), matrix.NewCoord(Input, 2*Input),
		matrix.NewCoord(Input, Input), matrix.NewCoord(Input, 1))
	w, h := opt[0].Output.Output.W, opt[0].Output.Output.H
	var sample matrix.Sample
	for i := 0; i < 33; i++ {
		sample = optimizer.Iterate()
		fmt.Println(i, sample.Cost)
		cost := sample.Cost
		type Coord struct {
			Signal float32
			Coord  int
		}
		type Result struct {
			Color  byte
			Signal float32
			IX     int
			IY     int
			X      []Coord
			Y      []Coord
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
			x := make([]Coord, w)
			for j, value := range xx {
				x[j].Coord = j
				x[j].Signal = value
			}
			sort.Slice(x, func(i, j int) bool {
				return x[i].Signal > x[j].Signal
			})
			yy := w1.Data[offset+10+w : offset+10+w+h]
			y := make([]Coord, h)
			for j, value := range yy {
				y[j].Coord = j
				y[j].Signal = value
			}
			sort.Slice(y, func(i, j int) bool {
				return y[i].Signal > y[j].Signal
			})
			result := Result{
				Color:  byte(color),
				Signal: maxColor,
				IX:     0,
				IY:     0,
				X:      x,
				Y:      y,
			}

			var apply func(result Result) bool
			apply = func(result Result) bool {
				x, y := result.X[result.IX].Coord, result.Y[result.IY].Coord
				if result.Signal > grid[y][x].Signal {
					if grid[y][x].Signal != 0 {
						for {
							sx := false
							if grid[y][x].IX < w-1 {
								sx = true
								grid[y][x].IX++
								if apply(grid[y][x]) {
									break
								}
							}
							sy := false
							if grid[y][x].IY < h-1 {
								sy = true
								grid[y][x].IY++
								if apply(grid[y][x]) {
									break
								}
							}
							if sx && sy {
								break
							}
						}
					}
					grid[y][x] = result
					return true
				}
				return false
			}
			for {
				sx := false
				if result.IX < w-1 {
					sx = true
					result.IX++
					if apply(result) {
						break
					}
				}
				sy := false
				if result.IY < h-1 {
					sy = true
					result.IY++
					if apply(result) {
						break
					}
				}
				if sx && sy {
					break
				}
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
