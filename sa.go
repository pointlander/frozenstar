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
	sets := Load()
	get := func(s, example int) matrix.Matrix {
		pairs := make([]Pair, 0, 8)
		length := 0
		set := sets[s]
		t := set.Train[example]
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
				length++
			}
		}
		for j, v := range t.Output {
			for i := range v {
				pair.Output = append(pair.Output, Pixel{
					C: v[i],
					X: i,
					Y: j,
				})
				length++
			}
		}
		pairs = append(pairs, pair)

		in := matrix.NewMatrix(Input, length)
		for _, pair := range pairs {
			for _, p := range pair.Input {
				input := matrix.NewZeroMatrix(Input, 1)
				input.Data[p.C] = 1
				input.Data[10+p.X] = 1
				input.Data[10+30+p.Y] = 1
				in.Data = append(in.Data, input.Data...)
			}
			for _, p := range pair.Output {
				input := matrix.NewZeroMatrix(Input, 1)
				input.Data[p.C] = 1
				input.Data[10+p.X] = 1
				input.Data[10+30+p.Y] = 1
				input.Data[10+30+30] = 1
				in.Data = append(in.Data, input.Data...)
			}
		}
		return in
	}
	in := get(0, 0)
	fmt.Println(in.Cols, in.Rows)
	entropy := matrix.SelfEntropy64(in, in, in)
	fmt.Println(len(entropy), entropy)
}
