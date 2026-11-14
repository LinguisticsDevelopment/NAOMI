package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
)

var consumption []bool
var words []*Node
var length int
var edges []*Connection

var assignments map[string]map[string]map[string]string

func ParserInit() {
	jsonFile, err := os.Open("assignments.json")
	if err != nil {
		fmt.Println(err)
	}

	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	json.Unmarshal(byteValue, &assignments)
}

type ConFrame struct {
	parent     *Node
	child      *Node
	childIndex int
	conType    ConnectionType
	consume    bool
}

func SentenceParse(w []*Node) (*Node, error) {
	words = w
	consumption = make([]bool, len(words))
	length = len(words)

	ruleSet := ruleSetParse("english_rules.json") //TODO: THIS IS WHERE LANGUAGE IS CHOSEN
	for _, rule := range ruleSet {
		for i, word := range words {
			fmt.Fprintf(os.Stderr, "%s %s %t, ", word.Value.Text, NodeTypeToString[word.Type], consumption[i])
		}
		fmt.Fprintf(os.Stderr, "\n")
		iterativeParse(rule)
	}

	func() {}()

	first := -1
	count := 0

	for index, _ := range words {
		if !consumption[index] {
			first = index
			// PrintGraph(Web{nil, words[index]})
			count += 1
		}
	}

	if count == 0 {
		return nil, errors.New("Could not find root node in SentenceParse")
	}
	if count > 1 {
		return nil, errors.New("Multiple root nodes in SentenceParse")
	}

	return words[first], nil
}

var pullTypes = map[SubCat]SubType{}

func iterativeParse(rules []Rule) {
	var connectionQueue []ConFrame
	for wordIndex := 0; wordIndex < length; wordIndex++ {
		if consumption[wordIndex] {
			continue
		}
		currentWord := words[wordIndex]
		connectionQueue = []ConFrame{}
		//for _, rule := range rules {

		for ruleNum := 0; ruleNum < len(rules); ruleNum++ {
			pullTypes = map[SubCat]SubType{}
			rule := rules[ruleNum]
			fail := false
			//Check rule, if -1 then continue
			//add parts
			if rule.RootType == currentWord.Type {
				for _, part := range rule.Before {
					newBefores := getAmount(part, -1, wordIndex, rule.PullCats)
					//add new connection to queue
					if newBefores[0] == -1 {
						fail = true
						break
					}
					for _, before := range newBefores {
						connType := StringToConnectionType[assignments[NodeTypeToString[words[before].Type]]["before"][NodeTypeToString[currentWord.Type]]]
						connection := ConFrame{currentWord, words[before], before, connType, !part.SkipConsumption}
						//modify Subtypes
						connectionQueue = append(connectionQueue, connection)
					}
				}
				if fail {
					continue
				}
				for _, part := range rule.After {
					newAfters := getAmount(part, 1, wordIndex, rule.PullCats)
					//add new connection to queue
					if newAfters[0] == -1 {
						fail = true
						break
					}
					for _, after := range newAfters {
						connType := StringToConnectionType[assignments[NodeTypeToString[words[after].Type]]["after"][NodeTypeToString[currentWord.Type]]]
						connection := ConFrame{currentWord, words[after], after, connType, !part.SkipConsumption}
						//modify Subtypes
						connectionQueue = append(connectionQueue, connection)
					}
				}
				if fail {
					continue
				}
			} else {
				continue
			}
			if fail {
				continue
			}
			for _, connection := range connectionQueue {
				skip := false
				for _, edge := range edges {
					if edge.Type == connection.conType && edge.A == connection.parent && edge.B == connection.child {
						skip = true
					}
				}
				if !skip {
					c := Connect(connection.conType, connection.parent, connection.child)
					edges = append(edges, c)
				}
				//if consumption is true:
				if connection.consume {
					consumption[connection.childIndex] = true
				}
			}
			currentWord.Type = rule.Resultant
			currentWord.Flags = []SubType{}
			for _, sub := range pullTypes {
				currentWord.Flags = append(currentWord.Flags, sub)
			}
			if rule.IsRecursive {
				ruleNum = 0
			} else {
				break
			}

		}

		//if recursive, wordIndex-1
	}
}

func getUnconsumed(dir int, part NodeType, subs []SubType, cats []SubCat, index int, root int, pull []SubCat, og bool) int {
	next := index + dir
	if next < 0 || next >= length {
		return -1
	}
	for consumption[next] {
		next += dir
		// if next+dir < 0 || next+dir >= length {
		// 	return -1
		// }
		if next < 0 || next >= length {
			return -1
		}
	}

	if (words[next].Type != part) && !og {
		return -1
	} else if (words[next].OG != part) && og {
		return -1
	}

	//grab root sub of cat type
	//check test sub of cat type

	for _, cat := range cats { //WOOOO CATEGORY MATCHING
		rootSub := getSubFromCat(cat, root)
		if rootSub == -1 {
			return -1
		}
		if rootSub != getSubFromCat(cat, next) {
			return -1
		}
	}

	for _, sub := range subs {
		if checkSubtype(sub, next) < 0 {
			return -1
		}
	}

	//Check for subtype and subCat matching here

	for _, cat := range pull {
		subAdd := getSubFromCat(cat, next)
		pullTypes[cat] = subAdd
	}

	return next
}

func getAll(dir int, part NodeType, subs []SubType, cats []SubCat, index int, pull []SubCat, og bool) []int {
	indices := []int{}
	next := getUnconsumed(dir, part, subs, cats, index, index, pull, og)
	if next == -1 { /*  */
		return []int{-1}
	}
	for next != -1 {
		indices = append(indices, next)
		next = getUnconsumed(dir, part, subs, cats, next, index, pull, og)
	}

	return indices
}

// func getAmount(amount int, dir int, gap int, part Tag, index int) []int {
func getAmount(part Part, dir, index int, pull []SubCat) []int {
	//This is a super helper function. If you want the next unconsumed,
	//you can input 1. If you want all of the unconsumed in that direction,
	//you can put -1. Plan to implement ability to shift over start point for
	//gap searches (think verbs and various types of objects)
	match := part.TypeKind
	subs := part.SubTypes
	cats := part.SubCats
	og := part.CheckOG
	// gap := part.Distance
	// indices := getAll(dir, match, index+gap)
	indices := getAll(dir, match, subs, cats, index, pull, og)
	if part.FindAllinDir {
		return indices
	} else {
		return indices[0:1]
	}
	// slice of indices depending on distance from start and number in chain
}

func checkSubtype(sub SubType, index int) int {
	for _, flag := range words[index].Flags {
		if flag == sub {
			return -1
		}
	}
	return 1
}

func getSubFromCat(cat SubCat, index int) SubType {
	for _, flag := range words[index].Flags {
		if SubTypeToSubCat[flag] == cat {
			return flag
		}
	}
	return -1
}
