package main

import (
	"fmt"
	"os"
)

func main() {

	// web, err := NewWeb("")
	// web, err := NewWeb("The tall boy is trucks, cars, and trains.")
	// web, err := NewWeb("The tall boy is trucks, cars, and trains.")
	// web, err := NewWeb("Bob likes Jane.")
	// web, err := NewWeb("The car is red.")
	// web, err := NewWeb("Luke and Will play basketball and code.")
	// web, err := NewWeb("Luke and Will code and play basketball.")
	// web, err := NewWeb("Bob and Carl kill and rob.")
	// web, err := NewWeb("Taking the derivative of a function is easy.")
	// web, err := NewWeb("The derivative of a function is easy.")
	// web, err := NewWeb("The boy goes under the table.")
	// web, err := NewWeb("Bob is short, fat, dumb, evil, and french.")
	// web, err := NewWeb("Bob is short, fat, dumb, evil, and french.")

	// web, err := NewWeb("What is the boy.")

	// web, err := NewWeb("The tree is green and the cat is orange.")
	// NewWeb("The tree is green and the cat is orange.")

	// web, err := NewWeb("The tree is green, the cat is orange, and the ocean is blue.")
	// NewWeb("The tree is green, the cat is orange, and the ocean is blue.")

	// web, err := NewWeb("The house of the duke of the shire of the mannor of the country.")

	// web, err := NewWeb("The boy is tall and red.")

	// if err != nil {
	// 	fmt.Fprintf(os.Stderr, "%v\n", err)
	// } else {
	// 	PrintWeb(web)
	// }

	// ---------------------------------------- DeepCopyTest
	// web, err := NewWeb("The tall boy is trucks, cars, and trains.")
	// if err != nil {
	// 	fmt.Fprintf(os.Stderr, "%v\n", err)
	// }

	// copy := DeepCopyWeb(&web)
	// fmt.Fprintf(os.Stderr, "%s\n", copy.Root.Value.Text)
	// PrintWeb(web)

	// ---------------------------------------- SplitTest
	// web, err := NewWeb("The tall boy is trucks, cars, and trains.")
	// if err != nil {
	// 	fmt.Fprintf(os.Stderr, "%v\n", err)
	// }

	// res := SplitWebAtCoord(web)
	// fmt.Fprintf(os.Stderr, "Length: %d\n", len(res))
	// for i, w := range res {
	// 	SimplePrintWeb(w)
	// 	if i == 2 {
	// 		// PrintWeb(*w)
	// 	}
	// }

	// ---------------------------------------- CompareWebTest
	// web, err := NewWeb("The boy is tall.")
	// if err != nil {
	// 	fmt.Fprintf(os.Stderr, "%v\n", err)
	// }
	// context, err := NewWeb("The boy is tall and red.")

	// result := web.CompareWeb(&context)
	// fmt.Fprintf(os.Stderr, "Result %t\n", result)

	// if err != nil {
	// 	fmt.Fprintf(os.Stderr, "%v\n", err)
	// } else {
	// 	PrintWeb(web)
	// }

	// ---------------------------------------- CompareWebsTest
	web, err := NewWeb("The tall boy is trucks.")
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
	}

	context, err := NewWeb("The tall boy is trucks, cars, and trains.")
	res := SplitWebAtCoord(context)

	result := web.CompareWebs(res)
	fmt.Fprintf(os.Stderr, "Result %t\n", result)

	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
	} else {
		PrintWeb(web)
	}
}

/*
	likes := NewNode(N_VERBAL, "like", POS_VERB)
	boy := NewNode(N_NOMINAL, "boy", POS_NOUN)
	the := NewNode(N_DESCRIPTOR, "the", POS_DET)
	tall := NewNode(N_DESCRIPTOR, "tall", POS_DET)
	and := NewNode(N_NOMINAL, "and", POS_CCONJ)
	trucks := NewNode(N_NOMINAL, "truck", POS_NOUN)
	cars := NewNode(N_NOMINAL, "car", POS_NOUN)
	trains := NewNode(N_NOMINAL, "train", POS_NOUN)

	//A is the Upper Node

	Connect(C_SUBJECT, likes, boy)

	Connect(C_DESCRIPTION, boy, the)
	Connect(C_DESCRIPTION, boy, tall)

	Connect(C_OBJECT, likes, and)

	Connect(C_COORDINATION, and, trucks)
	Connect(C_COORDINATION, and, cars)
	Connect(C_COORDINATION, and, trains)

	root := likes
	graph := Web{nil, root}

	PrintGraph(graph)

*/
