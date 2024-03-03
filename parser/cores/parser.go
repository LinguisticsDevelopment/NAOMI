// Sabrina - Shifting After-Before Ruleset Iterative Node Allocator
package cores

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	. "naomi/parser/cores/words"
	"os"

	"github.com/oleiade/reflections"
)

//var count int = 0 //Debug purposes

func SNA(sentencePointers []*Word) (*Word, bool) { //Shifting Node Allocator
	reduced := sentencePointers
	for coordIndex := len(reduced) - 1; coordIndex >= 0; coordIndex-- {
		coordWord := reduced[coordIndex]
		if coordWord.Function == "coord" {
			listOfWords := make([]Word, len(reduced))
			for addIndex := range reduced {
				name := reduced[addIndex].Name
				part := reduced[addIndex].Part
				listOfWords[addIndex] = BuildWord(name, part)
			}
			firstClause := []*Word{}
			secondClause := []*Word{}
			for index := 0; index < len(listOfWords); index++ {
				if index < coordIndex {
					firstClause = append(firstClause, &listOfWords[index])
				} else if index > coordIndex {
					secondClause = append(secondClause, &listOfWords[index])
				}
			}
			//fmt.Print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
			_, firstBreaks := ABRI(firstClause)
			//fmt.Println("\nPaired With:")
			_, secondBreaks := ABRI(secondClause)
			//fmt.Print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
			if !secondBreaks && !firstBreaks {
				Clause1, _ := ABRI(reduced[:coordIndex])
				Clause2, _ := ABRI(reduced[coordIndex+1:])
				ApplyAlong(coordWord, Clause1, "before", false)
				ApplyAlong(coordWord, Clause2, "after", false)
				coordWord.Function = "clause"
				return coordWord, false
			}
		}
	}
	finalWord, _ := ABRI(reduced)
	return finalWord, false
}

func ABRI(sentencePointers []*Word) (*Word, bool) { //After-Before Ruleset Iteration Parser
	ruleSet := ruleSetParse("rules.json")
	reduced := sentencePointers

	for subIndex, subWord := range reduced {
		if subWord.Function == "suboord" {
			trueEndIndex := len(reduced)
			for endIndex := len(reduced); endIndex > subIndex; endIndex-- {
				listOfWords := make([]Word, len(reduced))
				for addIndex := range reduced {
					name := reduced[addIndex].Name
					part := reduced[addIndex].Part
					listOfWords[addIndex] = BuildWord(name, part)
				}

				listOfWords[subIndex] = BuildWord("tempNoun", "noun")

				mainClause := []*Word{}
				subClause := []*Word{}
				for index := 0; index < len(listOfWords); index++ {
					if index < subIndex {
						mainClause = append(mainClause, &listOfWords[index])
					} else if index >= endIndex {
						mainClause = append(mainClause, &listOfWords[index])
					} else {
						subClause = append(subClause, &listOfWords[index])
					}
				}
				//fmt.Print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
				_, firstBreaks := ABRI(subClause)
				//fmt.Println("\nPaired With:")
				_, secondBreaks := ABRI(mainClause)
				//fmt.Print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
				if !secondBreaks && !firstBreaks {
					trueEndIndex = endIndex
					break
				} else if endIndex == subIndex {
					return &Word{}, true
				}
			}
			startIndex := subIndex + 1
			tempNoun := []*Word{}
			temp := BuildWord("temp", "noun")
			tempNoun = append(tempNoun, &temp)
			clauseBounded := reduced[startIndex:trueEndIndex]
			clauseBounded = append(tempNoun, clauseBounded...)
			newClause, _ := ABRI(clauseBounded)
			newClause.Consumed = true
			for refIndex := range clauseBounded {
				clauseBounded[refIndex].Consumed = true
			}
			ApplyAlong(subWord, newClause, "after", true)
			subWord.Function = "suboordClause"
		}
	}
	newReduced := []*Word{}
	for _, word := range reduced {
		if !word.Consumed {
			newReduced = append(newReduced, word)
		}
	}
	reduced = newReduced

	for rule := 0; rule < len(ruleSet); rule++ {

		reduced = IterativeParse(reduced, ruleSet[rule])
	}

	newReduced = []*Word{}
	for _, word := range reduced {
		if !word.Consumed {
			newReduced = append(newReduced, word)
		}
	}
	reduced = newReduced

	if len(reduced) == 1 && reduced[0].Function == "clause" {
		return reduced[0], false
	} else {
		return &Word{}, true
	}
}

// Fix needed is a subclass checker or else i will assign everything to the wrong types
func IterativeParse(working []*Word, rules []Rule) []*Word {
	//output := []*Word{}
	for wordIndex := 0; wordIndex < len(working); wordIndex++ {
		currentWord := working[wordIndex]
		if !currentWord.Consumed && !currentWord.Suboordinated {
			for ruleNum := 0; ruleNum < len(rules); ruleNum++ {
				currentRule := rules[ruleNum]
				if currentWord.Function == currentRule.RootType {
					works := true
					type applied struct {
						apply        *Word
						skipConsumed bool
						index        int
					}
					OverSubType := ""
					befores := []applied{}
					afters := []applied{}
					for _, before := range currentRule.Before {
						newApp := []applied{}

						lastIndex := wordIndex
						if len(befores) > 0 {
							lastIndex = befores[len(befores)-1].index
						}
						workingWord, nextInt := GetUnconsumed(lastIndex, -1, working)
						newApp = append(newApp, applied{apply: workingWord, skipConsumed: before.SkipConsumption, index: nextInt})

						if workingWord.Function != before.TypeKind {
							works = false
						}
						if before.SubType != "" {
							if workingWord.SubType != before.SubType {
								works = false
							}
						}
						if currentRule.NeedsMatching {
							OverSubType = workingWord.SubType
						}
						if before.FindAllinDir {
							newVals := GetUntil(workingWord.Function, nextInt, -1, working)
							for _, ref := range newVals {
								newApplier := applied{apply: ref.word, skipConsumed: before.SkipConsumption, index: ref.index}
								newApp = append(newApp, newApplier)
							}
						}
						befores = append(befores, newApp...)
					}

					for _, after := range currentRule.After {
						newApp := []applied{}
						lastIndex := wordIndex
						if len(afters) > 0 {
							lastIndex = afters[len(afters)-1].index
						}
						workingWord, nextInt := GetUnconsumed(lastIndex, 1, working)
						newApp = append(newApp, applied{apply: workingWord, skipConsumed: after.SkipConsumption, index: nextInt})

						if workingWord.Function != after.TypeKind {
							works = false
						}
						if after.SubType != "" {
							if workingWord.SubType != after.SubType {
								works = false
							}
						}
						if after.FindAllinDir {
							newVals := GetUntil(workingWord.Function, nextInt, 1, working)
							for _, ref := range newVals {
								newApplier := applied{apply: ref.word, skipConsumed: after.SkipConsumption, index: ref.index}
								newApp = append(newApp, newApplier)
							}
						}
						//final := workingWord
						if currentRule.NeedsMatching {
							if workingWord.SubType != OverSubType {
								works = false
							}
						}

						afters = append(afters, newApp...)
					}

					if currentRule.RootSubType != "" {
						if currentWord.SubType != currentRule.RootSubType {
							works = false
						}
					}
					if works {
						if currentRule.NeedsMatching {
							currentWord.SubType = OverSubType
						}
						//go through before and afters and then apply all words to current along their axises indicated in their word
						for _, beforeWord := range befores {
							ApplyAlong(currentWord, beforeWord.apply, "before", beforeWord.skipConsumed)
						}
						for _, afterWord := range afters {
							ApplyAlong(currentWord, afterWord.apply, "after", afterWord.skipConsumed)
						}
						currentWord.Function = currentRule.Resultant
						if currentRule.IsRecursive {
							ruleNum = 0
						} else {
							break
						}
					}
				}
			}
			//output = append(output, currentWord)
		}
	}

	//if rules[0].Resultant == "noun" && len(rules) == 1 {
	//	count++
	//	fmt.Println("\n", count, "~~~~~~", rules[0].Resultant)
	//	for _, word := range working {
	//		fmt.Println(word.Name, word.Part, word.Function, word.Consumed)
	//	}
	//}

	trueOutput := []*Word{}
	for _, word := range working {
		if !word.Consumed {
			trueOutput = append(trueOutput, word)
		}
	}
	return trueOutput
}

func GetUnconsumed(startIndex int, direction int, sentence []*Word) (*Word, int) { //pass in +1 or -1 for direction
	for wordIndex := startIndex + direction; len(sentence) > wordIndex && wordIndex >= 0; wordIndex = wordIndex + direction {
		nextWord := sentence[wordIndex]
		if !nextWord.Consumed && !nextWord.Suboordinated {
			return nextWord, wordIndex
		}
	}
	return &Word{}, 0
}

func GetUntil(funct string, startIndex int, direction int, sentence []*Word) []struct {
	word  *Word
	index int
} { //Get (All) Until (Not), pass in +1 or -1 for direction

	catches := []struct {
		word  *Word
		index int
	}{}
	for wordIndex := startIndex + direction; len(sentence) > wordIndex && wordIndex >= 0; wordIndex = wordIndex + direction {
		nextWord := sentence[wordIndex]
		if !nextWord.Consumed && !nextWord.Suboordinated {
			if nextWord.Function == funct {
				newPart := struct {
					word  *Word
					index int
				}{word: nextWord, index: wordIndex}
				catches = append(catches, newPart)
			} else {
				break
			}
		}
	}
	return catches
}

func ApplyAlong(receiver *Word, applier *Word, direction string, skipConsumption bool) {
	jsonFile, err := os.Open("assignments.json")
	if err != nil {
		fmt.Println(err)
	}

	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	var assignSet map[string]map[string]map[string]string

	json.Unmarshal(byteValue, &assignSet)

	applyField := assignSet[applier.Function][direction][receiver.Function]
	parts, _ := reflections.GetField(receiver, applyField) //reflections is the problem?

	switch parts := parts.(type) {
	case []*Word:
		parts = append(parts, applier)
		reflections.SetField(receiver, applyField, parts)
	case map[*Word]*Word:
		parts[applier] = applier.Connected
		reflections.SetField(receiver, applyField, parts)
	case *Word:
		reflections.SetField(receiver, applyField, applier)
	}

	if !skipConsumption {
		applier.Consumed = true
	}
}
