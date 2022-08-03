package client

import (
	"../common"
)

func thoughtProccess(root common.Thought) []common.Thought {
	rootThoughts := []common.Thought{root}
	checkingDISJUNCTIONs := true
	for checkingDISJUNCTIONs { //Will do this loop until there is no more DISJUNCTIONs
		newThoughts := []common.Thought{}
		hasDISJUNCTION := false
		for _, thisThought := range rootThoughts {
			if thisThought.Form == common.DISJUNCTION {
				for _, alt := range thisThought.Aspects[common.POSSIBILITIES] {
					newThoughts = append(newThoughts, *alt)
				}
				hasDISJUNCTION = true
			} else {
				thisExcluded := false
				for aspect, list := range thisThought.Aspects {
					if len(list) != 0 { //this check may be redundant
						for index, portion := range list {
							resultThoughts, isDISJUNCTION := splitOrs(*portion, thisThought, aspect, index)
							if isDISJUNCTION {
								for _, portion := range resultThoughts {
									newThoughts = append(newThoughts, portion)
								}
								thisExcluded = true
								hasDISJUNCTION = true
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
		if !hasDISJUNCTION {
			checkingDISJUNCTIONs = false
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
	if current.Form == common.DISJUNCTION {
		for _, alt := range current.Aspects[common.POSSIBILITIES] {
			root.Aspects[aspect][index] = alt
			multipleThoughts = append(multipleThoughts, root)
		}
		return multipleThoughts, true
	} else {
		hasDISJUNCTION := false
		for aspect, list := range current.Aspects {
			if len(list) != 0 { //this check may be redundant
				for index, portion := range list {
					resultThoughts, isDISJUNCTION := splitOrs(*portion, current, aspect, index)
					if isDISJUNCTION {
						for _, portion := range resultThoughts {
							multipleThoughts = append(multipleThoughts, portion)
						}
						hasDISJUNCTION = true
						break
					}
				}
			}
			if hasDISJUNCTION {
				break
			}
		}
		if hasDISJUNCTION {
			return multipleThoughts, true
		} else {
			return []common.Thought{root}, false
		}
	}
}

func Thinking(current *common.Thought) *common.Thought { //THIS SHOULD ONLY TAKE IN THOUGHTS THAT DO NOT HAVE ANY DISJUNCTIONS

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
	case common.CONJUNCTION:
		doCONJUNCTIONs(current)
	case common.DEVELOPMENT:
		doDEVELOPMENTs(current)
	case common.CONDITION:
		doCONDITIONs(current)
	case common.IMPLICATION:
		doIMPLICATIONs(current)
	case common.CONCEPT:
		doConcepts(current)
	}
	return current
}

func doCONJUNCTIONs(current *common.Thought) *common.Thought {
	for _, part := range current.Aspects[common.COMPONENT] { //remember, all parts should be proccessed by now
		if part.Form == common.CONJUNCTION {
			for _, subPart := range part.Aspects[common.COMPONENT] {
				current.Aspects[common.COMPONENT] = append(current.Aspects[common.COMPONENT], subPart)
				//Takes all parts within the parts of part if the part is another "CONJUNCTION" and adds them to current
			}
		}
	}
	return current
}

func doDEVELOPMENTs(current *common.Thought) *common.Thought {
	for _, part := range current.Aspects[common.SUBJECT] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.CONJUNCTION:
			//idk
		case common.CONCEPT:
			//idk
		}
	}
	for _, part := range current.Aspects[common.OBJECT] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.CONJUNCTION:
			//idk
		case common.CONCEPT:
			//idk
		}
	}
	for _, part := range current.Aspects[common.INDIRECT] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.CONJUNCTION:
			//idk
		case common.CONCEPT:
			//idk
		}
	}
	return current
}

func doCONDITIONs(current *common.Thought) *common.Thought {
	for _, part := range current.Aspects[common.SUBJECT] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.CONJUNCTION:
			//idk
		case common.CONCEPT:
			//idk
		}
	}
	for _, part := range current.Aspects[common.COMPLEMENT] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.CONJUNCTION: //Multiple Adverbs, Adjectives, and Prepositions
			//idk
		case common.CONCEPT: //Should be Adverbs and Adjectives
			//idk
		}
	}
	return current
}

func doIMPLICATIONs(current *common.Thought) *common.Thought {
	for _, part := range current.Aspects[common.PARAMETER] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.CONJUNCTION:
			//idk
		case common.DEVELOPMENT:
			//idk
		case common.CONDITION:
			//idk
		case common.IMPLICATION:
			//idk
		case common.CONCEPT:
			//idk

		}
	}
	for _, part := range current.Aspects[common.IF] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.CONJUNCTION:
			//idk
		case common.DEVELOPMENT:
			//idk
		case common.CONDITION:
			//idk
		case common.IMPLICATION:
			//idk
		case common.CONCEPT:
			//idk
		}
	}
	for _, part := range current.Aspects[common.ELSE] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.CONJUNCTION:
			//idk
		case common.DEVELOPMENT:
			//idk
		case common.CONDITION:
			//idk
		case common.IMPLICATION:
			//idk
		case common.CONCEPT:
			//idk
		}
	}
	return current
}

func doConcepts(current *common.Thought) *common.Thought {
	for _, part := range current.Aspects[common.MULTIPLIERS] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.CONJUNCTION:
			//idk
		case common.CONCEPT:
			//idk
		}
	}
	for _, part := range current.Aspects[common.ADDITIVES] { //remember, all parts should be proccessed by now
		switch part.Form {
		case common.CONJUNCTION:
			//idk
		case common.CONCEPT:
			//idk
		}
	}
	return current
}
