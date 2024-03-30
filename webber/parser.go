package main

var consumption []bool
var words []*Node
var length int
var edges []*Connection

type ConFrame struct {
	parent     *Node
	child      *Node
	childIndex int
	conType    ConnectionType
	consume    bool
}

func SentenceParse(w []*Node) {
	words = w
	consumption = make([]bool, len(words))
	length = len(words)

}

func iterativeParse(rules []Rule) {
	var connectionQueue []ConFrame
	for wordIndex := 0; wordIndex < length; wordIndex++ {
		if consumption[wordIndex] {
			continue
		}
		currentWord := words[wordIndex]
		connectionQueue = []ConFrame{}
		for _, rule := range rules {
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
				}
				if fail {
					continue
				}
			} else {
				continue
			}
			for _, connection := range connectionQueue {
				Connect(connection.conType, connection.parent, connection.child)
				//if consumption is true:
				if connection.consume {
					consumption[connection.childIndex] = true
				}

			}
		}
		//if recursive, wordIndex-1
	}
}

func buildConnection(frame ConFrame) {

}

func getUnconsumed(dir int, part NodeType, index int) int {
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

	if words[next].Type != part {
		return -1
	}

	return next
}

func getAll(dir int, part NodeType, index int) []int {
	indices := []int{}
	next := getUnconsumed(dir, part, index)
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
	gap := part.Distance
	indices := getAll(dir, match, index+gap)
	if part.FindAllinDir {
		return indices
	} else {
		return indices[0:1]
	}
	// slice of indices depending on distance from start and number in chain
}
