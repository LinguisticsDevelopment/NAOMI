package cores

import (
	"fmt"
	. "naomi/parser/cores/words"
)

func TotalParse(sentence []Word) {
	sentencePointers := []*Word{}
	for index := range sentence {
		sentencePointers = append(sentencePointers, &sentence[index])
	}
	FinalWord, err := SNA(sentencePointers)
	if err {
		fmt.Println("something broke")
	} else {
		printer(FinalWord, 0)
	}
}

func printer(node *Word, indexLevel int) {
	if node.Part == "coord" {
		node.Level = indexLevel + 1
		front := whiteSpaceGetter(node.Level)
		fmt.Printf(front+"[%s] - coord -- [%s]\n", node.Name, node.Function)
		fmt.Println(front + " Parts of Coord:")
		for _, word := range node.Complements {
			printer(word, node.Level)
		}
		if node.Function == "participle" || node.Function == "clause" || node.Function == "verb" {
			node.Level = indexLevel + 1
			front := whiteSpaceGetter(node.Level)
			//fmt.Printf(front+"[%s] - verb\n", node.Name)
			fmt.Println(front + "Subject:")
			subject := node.Subject
			if subject.Name != "" {
				printer(node.Subject, node.Level)
			}
			fmt.Println(front + "Objects:")
			for _, object := range node.Objects {
				if object.Name != "" {
					printer(object, node.Level)
				}

			}
			fmt.Println(front + "SubComp:")
			subComp := node.SubjectComplement
			if subComp.Name != "" {
				printer(subComp, node.Level)

			}
			//fmt.Println(node.Name)

		}
	} else if node.Part == "suboord" {
		node.Level = indexLevel + 1
		front := whiteSpaceGetter(node.Level)
		fmt.Printf(front+"[%s] - suboord\n", node.Name)
		fmt.Println(front + " Start of Suboord Clause:")
		for _, word := range node.Suboordinations {
			if word.Function == "verb" {
				printer(word, indexLevel)
			} else {
				printer(word, node.Level)
			}

		}
		fmt.Println(front + " End of Suboord Clause")

	} else if node.Part == "prep" {
		node.Level = indexLevel
		front := whiteSpaceGetter(node.Level)
		fmt.Printf(front+"[%s] - prep\n", node.Name)
		//doPrepThings

	} else if node.Part == "verb" {
		node.Level = indexLevel + 1
		front := whiteSpaceGetter(node.Level)
		fmt.Printf(front+"[%s] - verb\n", node.Name)
		fmt.Println(front + "Subject:")
		subject := node.Subject
		if subject.Name != "" {
			printer(node.Subject, node.Level)
		}
		fmt.Println(front + "Objects:")
		for _, object := range node.Objects {
			if object.Name != "" {
				printer(object, node.Level)
			}

		}
		fmt.Println(front + "SubComp:")
		subComp := node.SubjectComplement
		if subComp.Name != "" {
			printer(subComp, node.Level)

		}
		//fmt.Println(node.Name)

	} else if node.Part == "noun" {
		node.Level = indexLevel + 1
		front := whiteSpaceGetter(node.Level)
		fmt.Printf(front+"[%s] - noun\n", node.Name)
		fmt.Println(front + " Descriptors of noun:")
		for _, adj := range node.Descriptors {
			printer(adj, node.Level)

		}
		//fmt.Println(node.Name)

	} else if node.Part == "adj" {
		node.Level = indexLevel + 1
		front := whiteSpaceGetter(node.Level)
		fmt.Printf(front+"[%s] - adj\n", node.Name)
		fmt.Println(front + " Specifiers of adj:")
		for _, adv := range node.Specifiers {
			printer(adv, node.Level)
		}
		//fmt.Println(node.Name)
	} else if node.Part == "adv" {
		node.Level = indexLevel + 1
		front := whiteSpaceGetter(node.Level)
		fmt.Printf(front+"[%s] - adj\n", node.Name)
		fmt.Println(front + " Specifiers of adv:")
		for _, adv := range node.Specifiers {
			printer(adv, node.Level)
		}
		//fmt.Println(node.Name)
	}
	if len(node.Connections) > 0 {
		node.Level = indexLevel + 1
		front := whiteSpaceGetter(node.Level)
		fmt.Println(front + " Connections:")
		for connector, connectedWord := range node.Connections {
			fmt.Println(front + "  To:" + connector.Name)
			printer(connectedWord, node.Level)
			//fmt.Println(front + "  Via:")
			//printer(connector, node.Level)
		}
	}
}

func whiteSpaceGetter(counts int) string {
	doubled := counts * 2
	returnSpace := ""
	for x := 0; x <= doubled; x++ {
		returnSpace = returnSpace + "  "
	}
	return returnSpace
}
