package main

import (
	"naomi/webber/parser"
)

var consumption []bool
var words []*Node
var length int
var edges []*Connection

type ConFrame struct {
	parent     *Node
	child      *Node
	childIndex int
	conType    ConnectionType
}

func SentenceParse(w []*Node) {

	words = w
	consumption = make([]bool, len(words))
	length = len(words)

}

func iterativeParse(rules []parser.Rule) {
	index := 0
	var connectionQueue []ConFrame
	for wordIndex := 0; wordIndex < length; wordIndex++ {
		if consumption[wordIndex] {
			continue
		}
		currentWord := words[wordIndex]
		connectionQueue = []ConFrame{}
		for _, rule := range rules {
			//Check rule, if -1 then continue
			//add parts
		}
		for _, connection := range connectionQueue {
			Connect(connection.conType, connection.parent, connection.child)
			//if consumption is true:
			consumption[connection.childIndex] = true
		}
		//if recursive, wordIndex-1
	}
}

func buildConnection(frame ConFrame) {

}

func getUnconsumed(dir int, part Tag, index int) int {
	next := dir + index
	if next < 0 || next >= length {
		return -1
	}
	for consumption[next] {
		next += dir
		if next+dir < 0 || next+dir >= length {
			return -1
		}
	}

	if words[next].Value.POS != part {
		return -1
	}

	return next
}

func getAll(dir int, part Tag, index int) []int {
	indices := []int{}
	next := getUnconsumed(dir, part, index)
	for next != -1 {
		indices = append(indices, next)
		next = getUnconsumed(dir, part, next)
	}

	return indices
}

func getAmount(amount int, dir int, gap int, part Tag, index int) []int {
	//This is a super helper function. If you want the next unconsumed,
	//you can input 1. If you want all of the unconsumed in that direction,
	//you can put -1. Plan to implement ability to shift over start point for
	//gap searches (think verbs and various types of objects)

	indices := get_all(dir, part, index+gap)
	return indices //slice of indices depending on distance from start and number in chain
}
