package words

//Connections - Prepositions
//Relationships - Possessors

func (receiver Meaning) RelativeConnect(director Meaning, donor Meaning) []Meaning {
	axisList := director.GetAxises() //get the Axises the two words are connected on
	for _, axisName := range axisList {
		recVal := 0.0
		donVal := 0.0
		if Contains(receiver.GetAxises(), axisName) {
			recVal = receiver.Assets[axisName].Relative
		}
		if Contains(donor.GetAxises(), axisName) {
			donVal = donor.Assets[axisName].Relative
		} else {
			donor.Assets[axisName] = Axis{Name: axisName, Relative: 0.0, Value: 0.0}
		}
		axis := receiver.Assets[axisName]
		directed := recVal + donVal + director.Assets[axisName].Relative
		receiver.Assets[axisName] = Axis{Value: axis.Value, Relative: directed, Name: axis.Name}
	}
	return []Meaning{receiver, donor}
}

func (pool Meaning) GetDifference(strainer Meaning) Meaning {
	diffMap := map[string]bool{}
	otherAxises := pool.GetAxises()
	for _, testAxis := range strainer.GetAxises() {
		diffMap[testAxis] = !Contains(otherAxises, testAxis)
	}
	allAxises := []Axis{}
	diffCount := 0
	for usedAxis, isDiff := range diffMap {
		if isDiff {
			allAxises[diffCount] = pool.Assets[usedAxis]
			diffCount++
		}
	}
	returnMeaning := Meaning{Name: "difference", Assets: map[string]Axis{}}
	for _, axis := range allAxises {
		returnMeaning.Assets[axis.Name] = axis
	}
	return returnMeaning //returns a meaning with the unique axises mapped to Assets and the name set to "difference"
}

func (receiver Word) Direct(director *Word) []Word {
	if director.Part == "connector" { //Prepositions are connectors, also includes "because" - These act as essentially adverbs
		for _, currentFunction := range director.prepFunc {
			directedSet := receiver.Scopes[currentFunction].RelativeConnect(director.Scopes[currentFunction], receiver.Connections[director].Scopes[currentFunction])
			receiver.Scopes[currentFunction] = directedSet[0]
			receiver.Connections[director].Scopes[currentFunction] = directedSet[1]
		}

	} else if director.Part == "superlater" { //Adjectives - prepositions that are relative on nominal axises
		for _, currentFunction := range director.prepFunc {
			directedSet := receiver.Nominals[currentFunction].RelativeConnect(director.Nominals[currentFunction], receiver.Connections[director].Nominals[currentFunction])
			receiver.Nominals[currentFunction] = directedSet[0]
			receiver.Connections[director].Nominals[currentFunction] = directedSet[1]
		}

	} else if director.Part == "kind" { //This means it is a relationship word like "kind" -> one consists of and recieves the aspects of the other
		donorUniques := receiver.Connections[director].Nominals["attributes"].GetDifference(receiver.Nominals["attributes"])
		receiver.Nominals["attributes"] = receiver.Nominals["attributes"].Administer(donorUniques, true)

	} else { //This means it is a relationship word like "part" -> one consists of and recieves the aspects of the other
		receiverUniques := receiver.Nominals["attributes"].GetDifference(receiver.Connections[director].Nominals["attributes"])
		receiver.Connections[director].Nominals["attributes"] = receiver.Connections[director].Nominals["attributes"].Administer(receiverUniques, true)

	}
	return []Word{receiver, *receiver.Connections[director]}
}
