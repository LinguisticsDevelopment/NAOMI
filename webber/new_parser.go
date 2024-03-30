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

	ruleSet := ruleSetParse("new_rules.json")
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
			PrintGraph(Web{nil, words[index]})
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
			rule := rules[ruleNum]
			fail := false
			//Check rule, if -1 then continue
			//add parts
			if rule.RootType == currentWord.Type {
				for _, part := range rule.Before {
					newBefores := getAmount(part, -1, wordIndex)
					//add new connection to queue
					if newBefores[0] == -1 {
						fail = true
						break
					}
					for _, before := range newBefores {
						connType := StringToConnectionType[assignments[NodeTypeToString[words[before].Type]]["before"][NodeTypeToString[currentWord.Type]]]
						connection := ConFrame{currentWord, words[before], before, connType, !part.SkipConsumption}
						connectionQueue = append(connectionQueue, connection)
					}
				}
				if fail {
					continue
				}
				for _, part := range rule.After {
					newAfters := getAmount(part, 1, wordIndex)
					//add new connection to queue
					if newAfters[0] == -1 {
						fail = true
						break
					}
					for _, after := range newAfters {
						connType := StringToConnectionType[assignments[NodeTypeToString[words[after].Type]]["after"][NodeTypeToString[currentWord.Type]]]
						connection := ConFrame{currentWord, words[after], after, connType, !part.SkipConsumption}
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
			if rule.IsRecursive {
				ruleNum = 0
			} else {
				break
			}

		}

		//if recursive, wordIndex-1
	}
}

func getUnconsumed(dir int, part NodeType, index int) int {
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

	if words[next].Type != part {
		return -1
	}

	return next
}

func getAll(dir int, part NodeType, index int) []int {
	indices := []int{}
	next := getUnconsumed(dir, part, index)
	if next == -1 {
		return []int{-1}
	}
	for next != -1 {
		indices = append(indices, next)
		next = getUnconsumed(dir, part, next)
	}

	return indices
}

// func getAmount(amount int, dir int, gap int, part Tag, index int) []int {
func getAmount(part Part, dir, index int) []int {
	//This is a super helper function. If you want the next unconsumed,
	//you can input 1. If you want all of the unconsumed in that direction,
	//you can put -1. Plan to implement ability to shift over start point for
	//gap searches (think verbs and various types of objects)
	match := part.TypeKind
	// gap := part.Distance
	// indices := getAll(dir, match, index+gap)
	indices := getAll(dir, match, index)
	if part.FindAllinDir {
		return indices
	} else {
		return indices[0:1]
	}
	// slice of indices depending on distance from start and number in chain
}
