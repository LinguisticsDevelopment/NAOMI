package client

import (
	"../common"
)

func thoughtProccess(root common.Thought) []common.Thought {
	rootThoughts := []common.Thought{root}
	checkingExclusions := true
	for checkingExclusions { //Will do this loop until there is no more Exclusions
		newThoughts := []common.Thought{}
		hasExclusion := false
		for _, thisThought := range rootThoughts {
			if thisThought.Form == common.EXCLUSION {
				for _, alt := range thisThought.Aspects[common.ALTERNATIVE] {
					newThoughts = append(newThoughts, *alt)
				}
				hasExclusion = true
			} else {
				thisExcluded := false
				for aspect, list := range thisThought.Aspects {
					if len(list) != 0 { //this check may be redundant
						for index, portion := range list {
							resultThoughts, isExclusion := splitOrs(*portion, thisThought, aspect, index)
							if isExclusion {
								for _, portion := range resultThoughts {
									newThoughts = append(newThoughts, portion)
								}
								thisExcluded = true
								hasExclusion = true
								break
							}
						}
					}
					if thisExcluded {
						break
					}
				}
			}
		}
		if !hasExclusion {
			checkingExclusions = false
		} else {
			rootThoughts = newThoughts
		}
	}
	//We now have a list of Exclusive Thoughts, with no or's inside of them
	//this list represents all of the possibilities of a thought chain
	//proccessing time
	for index := range rootThoughts {
		rootThoughts[index] = *Thinking(&rootThoughts[index])
	}
	return rootThoughts
}

func splitOrs(current common.Thought, root common.Thought, aspect int, index int) ([]common.Thought, bool) {
	multipleThoughts := []common.Thought{}
	if current.Form == common.EXCLUSION {
		for _, alt := range current.Aspects[common.ALTERNATIVE] {
			root.Aspects[aspect][index] = alt
			multipleThoughts = append(multipleThoughts, root)
		}
		return multipleThoughts, true
	} else {
		hasExclusion := false
		for aspect, list := range current.Aspects {
			if len(list) != 0 { //this check may be redundant
				for index, portion := range list {
					resultThoughts, isExclusion := splitOrs(*portion, current, aspect, index)
					if isExclusion {
						for _, portion := range resultThoughts {
							multipleThoughts = append(multipleThoughts, portion)
						}
						hasExclusion = true
						break
					}
				}
			}
			if hasExclusion {
				break
			}
		}
		if hasExclusion {
			return multipleThoughts, true
		} else {
			return []common.Thought{root}, false
		}
	}
}

func Thinking(current *common.Thought) *common.Thought { //THIS SHOULD ONLY TAKE IN THOUGHTS THAT DO NOT HAVE ANY EXCLUSIONS

	for aspect, list := range current.Aspects { //this makes sure we proccess all lower levels first
		if len(list) != 0 { //this check may be redundant
			for index, portion := range list {
				current.Aspects[aspect][index] = Thinking(portion)
			}
		}
	}
	return doThought(current)
}

func doThought(current *common.Thought) *common.Thought {
	switch current.Form {
	case common.INCLUSION:
		doInclusions(current)
		//Multiply the meanings across each other
	case common.CHANGE:
		doChanges(current)
		//Multiply the meanings across each other
	case common.STATE:
		doStates(current)
		//Multiply the meanings across each other
	case common.CONDITION:
		doConditions(current)
		//Multiply the meanings across each other
	case common.CONCEPT:
		doConcepts(current)
		//Multiply the meanings across each other
	case common.CONNECTION:
		doConnections(current)
		//I have no idea how and when to implement these now
	}
	return current
}

func doInclusions(current *common.Thought) *common.Thought {
	for _, part := range current.Aspects[common.COMPONENT] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.INCLUSION:
			//idk
		case common.CHANGE:
			//idk
		case common.STATE:
			//idk
		case common.CONDITION:
			//idk
		case common.CONCEPT:
			//idk
		case common.CONNECTION:
			//idk
		}
	}
	return current
}

func doChanges(current *common.Thought) *common.Thought {
	for _, part := range current.Aspects[common.SUBJECT] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.INCLUSION:
			//idk
		case common.CHANGE:
			//idk
		case common.STATE:
			//idk
		case common.CONDITION:
			//idk
		case common.CONCEPT:
			//idk
		case common.CONNECTION:
			//idk
		}
	}
	for _, part := range current.Aspects[common.OBJECT] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.INCLUSION:
			//idk
		case common.CHANGE:
			//idk
		case common.STATE:
			//idk
		case common.CONDITION:
			//idk
		case common.CONCEPT:
			//idk
		case common.CONNECTION:
			//idk
		}
	}
	for _, part := range current.Aspects[common.INDIRECT] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.INCLUSION:
			//idk
		case common.CHANGE:
			//idk
		case common.STATE:
			//idk
		case common.CONDITION:
			//idk
		case common.CONCEPT:
			//idk
		case common.CONNECTION:
			//idk
		}
	}
	return current
}

func doStates(current *common.Thought) *common.Thought {
	for _, part := range current.Aspects[common.SUBJECT] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.INCLUSION:
			//idk
		case common.CHANGE:
			//idk
		case common.STATE:
			//idk
		case common.CONDITION:
			//idk
		case common.CONCEPT:
			//idk
		case common.CONNECTION:
			//idk
		}
	}
	for _, part := range current.Aspects[common.COMPLEMENT] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.INCLUSION:
			//idk
		case common.CHANGE:
			//idk
		case common.STATE:
			//idk
		case common.CONDITION:
			//idk
		case common.CONCEPT:
			//idk
		case common.CONNECTION:
			//idk
		}
	}
	return current
}

func doConditions(current *common.Thought) *common.Thought {
	for _, part := range current.Aspects[common.PARAMETER] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.INCLUSION:
			//idk
		case common.CHANGE:
			//idk
		case common.STATE:
			//idk
		case common.CONDITION:
			//idk
		case common.CONCEPT:
			//idk
		case common.CONNECTION:
			//idk
		}
	}
	for _, part := range current.Aspects[common.IF] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.INCLUSION:
			//idk
		case common.CHANGE:
			//idk
		case common.STATE:
			//idk
		case common.CONDITION:
			//idk
		case common.CONCEPT:
			//idk
		case common.CONNECTION:
			//idk
		}
	}
	for _, part := range current.Aspects[common.ELSE] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.INCLUSION:
			//idk
		case common.CHANGE:
			//idk
		case common.STATE:
			//idk
		case common.CONDITION:
			//idk
		case common.CONCEPT:
			//idk
		case common.CONNECTION:
			//idk
		}
	}
	return current
}

func doConcepts(current *common.Thought) *common.Thought {
	for _, part := range current.Aspects[common.MULTIPLIERS] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.INCLUSION:
			//idk
		case common.CHANGE:
			//idk
		case common.STATE:
			//idk
		case common.CONDITION:
			//idk
		case common.CONCEPT:
			//idk
		case common.CONNECTION:
			//idk
		}
	}
	for _, part := range current.Aspects[common.ADDITIVES] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.INCLUSION:
			//idk
		case common.CHANGE:
			//idk
		case common.STATE:
			//idk
		case common.CONDITION:
			//idk
		case common.CONCEPT:
			//idk
		case common.CONNECTION:
			//idk
		}
	}
	return current
}

func doConnections(current *common.Thought) *common.Thought {
	for _, part := range current.Aspects[common.AFFECTED] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.INCLUSION:
			//idk
		case common.CHANGE:
			//idk
		case common.STATE:
			//idk
		case common.CONDITION:
			//idk
		case common.CONCEPT:
			//idk
		case common.CONNECTION:
			//idk
		}
	}
	for _, part := range current.Aspects[common.RELATED] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.INCLUSION:
			//idk
		case common.CHANGE:
			//idk
		case common.STATE:
			//idk
		case common.CONDITION:
			//idk
		case common.CONCEPT:
			//idk
		case common.CONNECTION:
			//idk
		}
	}
	return current
}
