// Vanessa - Vector-Abstraction, Natural Encoder, Sequential Summation Application
package cores

import (
	. "naomi/parser/cores/words"
	"sort"
	//"fmt"
)

type ByLevel []*Word

func (a ByLevel) Len() int           { return len(a) }
func (a ByLevel) Less(i, j int) bool { return a[i].Level < a[j].Level }
func (a ByLevel) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func RecursiveFlatten(node *Word) []*Word {
	Flattened := []*Word{}
	Flattened = append(Flattened, node.Subject)
	Flattened = append(Flattened, node.SubjectComplement)
	Flattened = append(Flattened, node.Objects...)
	Flattened = append(Flattened, node.Descriptors...)
	Flattened = append(Flattened, node.Specifiers...)
	Flattened = append(Flattened, node.Modifiers...)
	for _, relevant := range node.Connections {
		Flattened = append(Flattened, relevant)
	}
	Flattened = append(Flattened, node.Complements...)
	//for _, suboord := range node.Suboordinations {
	//	Masher(suboord)
	// it might be just one level back (-1)
	//}
	returnable := []*Word{}
	for _, word := range Flattened {
		word.Level = node.Level + 1
		returnable = append(returnable, RecursiveFlatten(word)...)
	}
	return returnable
}

func Masher(sentenceRoot *Word) *Word { //either returns a word pointer or a meaning pointer or a new structure altogether
	sentenceRoot.Level = 0
	//Final Level will = 0
	//Start level = organized[0]
	organized := RecursiveFlatten(sentenceRoot)
	sort.Sort(ByLevel(organized))
	maxLevel := organized[len(organized)-1].Level
	mappedWords := map[int][]*Word{}
	for LevelIndex := 0; LevelIndex <= maxLevel; LevelIndex++ {
		mappedWords[LevelIndex] = []*Word{}
	}
	for _, word := range organized {
		thisLevel := word.Level
		mappedWords[thisLevel] = append(mappedWords[thisLevel], word)
	}

	for currentIndex := maxLevel; currentIndex >= 0; currentIndex-- {
		for _, word := range mappedWords[currentIndex] {
			Abstractor(word)
		}
		for _, word := range mappedWords[currentIndex] {
			Connector(word)
		}
		for _, word := range mappedWords[currentIndex] {
			word = word
			//do the verb maths here; oh no
		}

	}

	return nil // TODO: Make not nil
}

func Abstractor(word *Word) {
	for _, specifier := range word.Specifiers {
		word.Specify(*specifier)
	}
	for _, descriptor := range word.Descriptors {
		word.Describe(*descriptor)
	}
	for _, modifier := range word.Modifiers {
		word.Modify(*modifier)
	}
	if word.Part == "coord" {
		for _, component := range word.Complements {
			for _, specifier := range word.Specifiers {
				component.Specify(*specifier)
			}
			for _, descriptor := range word.Descriptors {
				component.Describe(*descriptor)
			}
			for _, modifier := range word.Modifiers {
				component.Modify(*modifier)
			}
			for _, suboordinate := range word.Suboordinations {
				suboordinate = suboordinate
				//do the apply thingy with suboordinate - havent worked this out yet
			}
		}
	}
}

func Connector(word *Word) {
	for director, connected := range word.Connections {
		word.Direct(director)
		connected = connected
	}

	if word.Part == "coord" {
		for _, component := range word.Complements {
			for director, connected := range word.Connections {
				component.Direct(director)
				connected = connected
			}
		}
	}
}
func Suboordinations(word *Word) {
	for _, suboordinate := range word.Suboordinations {
		suboordinate.Subject = word
		Masher(suboordinate)
		word = word
	}

}
func FlowOFAction(word *Word) {
	// TODO: make work
	/*
		subjectRoleVector := word.GetVector("subject")
		subjectWordVector := word.Subject.GetVector("nominal")
		totalAxises := []string{}
		for _, component := range subjectVector.Direction {
			totalAxises = append(totalAxises, component.AxisName)
		}
		for _, axis := range subjectVector.Direction {

		}
		//do subject
		objectVector := word.GetVector("object")
		//do object
		indirectVector := word.GetVector("indirect")
		//do indirectObject
		complementVector := word.GetVector("complement")
		//do subjectComplement

		if word.Part == "coord" {
			word = word //coordinate word now needs to equal cross-product of components if and, then the or things, FANBOYS
		}
	*/
}
