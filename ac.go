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

// AC is an autocoder
func AC() {
	rng := matrix.Rand(1)

	sets := Load()
	done := make(chan bool, 8)
	process := func(sample *matrix.Sample) {
		opts := make([][]Opt, *FlagSets)
		for i := range opts {
			opts[i] = GetTrainingData(sets, i, 0)
		}
		x1 := sample.Vars[0][0].Sample()
		y1 := sample.Vars[0][1].Sample()
		z1 := sample.Vars[0][2].Sample()
		q := x1.Add(y1.H(z1))
		x2 := sample.Vars[1][0].Sample()
		y2 := sample.Vars[1][1].Sample()
		z2 := sample.Vars[1][2].Sample()
		k := x2.Add(y2.H(z2))
		x3 := sample.Vars[2][0].Sample()
		y3 := sample.Vars[2][1].Sample()
		z3 := sample.Vars[2][2].Sample()
		v := x3.Add(y3.H(z3))
		x4 := sample.Vars[3][0].Sample()
		y4 := sample.Vars[3][1].Sample()
		z4 := sample.Vars[3][2].Sample()
		w1 := x4.Add(y4.H(z4))
		x5 := sample.Vars[4][0].Sample()
		y5 := sample.Vars[4][1].Sample()
		z5 := sample.Vars[4][2].Sample()
		b1 := x5.Add(y5.H(z5))
		params := make([]matrix.Matrix, len(opts))
		for i := range params {
			x1 := sample.Vars[5+i][0].Sample()
			y1 := sample.Vars[5+i][1].Sample()
			z1 := sample.Vars[5+i][2].Sample()
			params[i] = x1.Add(y1.H(z1))
		}
		for i, opt := range opts {
			for j := range opt {
				offset := len(opt[j].Input.Input.I) + len(opt[j].Input.Output.I) + len(opt[j].Output.Input.I)
				for k := 0; k < params[i].Rows; k++ {
					offset := offset + k
					w1Offset := Input * k
					cc := opt[j].Opt.Data[Input*offset : Input*offset+10]
					w1CC := params[i].Data[w1Offset : w1Offset+10]
					maxCC, indexCC := float32(0.0), 0
					for key, value := range w1CC {
						if value > maxCC {
							indexCC, maxCC = key, value
						}
					}
					xx := opt[j].Opt.Data[Input*offset+10 : Input*offset+10+30]
					w1XX := params[i].Data[w1Offset+10 : w1Offset+10+30]
					maxXX, indexXX := float32(0.0), 0
					for key, value := range w1XX {
						if value > maxXX {
							indexXX, maxXX = key, value
						}
					}
					yy := opt[j].Opt.Data[Input*offset+10+30 : Input*offset+10+30+30]
					w1YY := params[i].Data[w1Offset+10+30 : w1Offset+10+30+30]
					maxYY, indexYY := float32(0.0), 0
					for key, value := range w1YY {
						if value > maxYY {
							indexYY, maxYY = key, value
						}
					}
					cc[indexCC] = 1
					xx[indexXX] = 1
					yy[indexYY] = 1
					opt[j].Opt.Data[Input*offset+10+30+30] = 1
				}
			}
		}
		sum := 0.0
		for _, opt := range opts {
			for i := range opt {
				output := w1.MulT(opt[i].Opt).Add(b1).Sigmoid()
				out := matrix.SelfAttention(q.MulT(output), k.MulT(output), v.MulT(output))
				for j := 0; j < out.Rows; j++ {
					for k := 0; k < out.Cols; k++ {
						diff := out.Data[j*out.Cols+k] - opt[i].Opt.Data[j*out.Cols+k]
						sum += float64(diff * diff)
					}
				}
			}
		}
		sample.Cost = sum
		done <- true
	}
	opts := make([][]Opt, *FlagSets)
	for i := range opts {
		opts[i] = GetTrainingData(sets, i, 0)
	}
	params := []matrix.Matrix{
		matrix.NewCoord(8*Input, Input), matrix.NewCoord(8*Input, Input), matrix.NewCoord(8*Input, Input),
		matrix.NewCoord(Input, 8*Input), matrix.NewCoord(8*Input, 1),
	}
	for _, opt := range opts {
		params = append(params, matrix.NewCoord(Input, opt[0].TargetSize()))
	}
	optimizer := matrix.NewOptimizer(&rng, 9, .1, 5+*FlagSets, func(samples []matrix.Sample, x ...matrix.Matrix) {
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
	}, params...)
	printResult := func(sample matrix.Sample, opt []Opt, m int) {
		w, h := opt[0].Output.Output.W, opt[0].Output.Output.H
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
		x1 := sample.Vars[m][0].Sample()
		y1 := sample.Vars[m][1].Sample()
		z1 := sample.Vars[m][2].Sample()
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
							if grid[y][x].IX < w {
								sx = true
								if apply(grid[y][x]) {
									break
								}
								grid[y][x].IX++
							}
							sy := false
							if grid[y][x].IY < h {
								sy = true
								if apply(grid[y][x]) {
									break
								}
								grid[y][x].IY++
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
				if result.IX < w {
					sx = true
					if apply(result) {
						break
					}
					result.IX++
				}
				sy := false
				if result.IY < h {
					sy = true
					if apply(result) {
						break
					}
					result.IY++
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
	}
	var sample matrix.Sample
	for i := 0; i < 33; i++ {
		sample = optimizer.Iterate()
		fmt.Println(i, sample.Cost)
		cost := sample.Cost
		for j, opt := range opts {
			printResult(sample, opt, 5+j)
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
