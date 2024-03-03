package words

//Descriptors - adjectives, numerals, determiners - Affect Nominals
//Specifiers - adverbs, negator, affirmer - Affect Scopes
//Modifiers - Modals & Auxilaries(Maybe) - Affect Roles or Scopes

func (receiver *Word) Describe(describer Word) { //Adjectives for Nouns
	description := describer.Nominals
	if receiver.ObjType == "representative" {
		for coordIndex, coordinate := range receiver.Complements {
			for _, meaningAxis := range coordinate.Nominals {
				receiver.Complements[coordIndex].Nominals[meaningAxis.Name] = meaningAxis.Administer(description[meaningAxis.Name], true)
			}
		}
	} else {
		for _, meaningAxis := range receiver.Nominals {
			receiver.Nominals[meaningAxis.Name] = meaningAxis.Administer(description[meaningAxis.Name], true)
		}
	}
}

func (receiver *Word) Specify(specifier Word) { //Adverbs for Adjectives, Adverbs, and Verbs
	description := specifier.Scopes
	if receiver.ObjType == "representative" {
		for coordIndex, coordinate := range receiver.Complements {
			for _, meaningAxis := range coordinate.Scopes {
				receiver.Complements[coordIndex].Scopes[meaningAxis.Name] = meaningAxis.Administer(description[meaningAxis.Name], false)
			}
		}
	} else {
		for _, meaningAxis := range receiver.Scopes {
			receiver.Scopes[meaningAxis.Name] = meaningAxis.Administer(description[meaningAxis.Name], false)
		}
	}
}

func (receiver *Word) Modify(modifier Word) {
	description := modifier.Roles
	if receiver.ObjType == "representative" {
		//For These Verbals Modifiers, There is going to need to be an indication of what meaning axis were working on, might just always apply to verbal fundementals
		for coordIndex, coordinate := range receiver.Complements {
			for _, meaningAxis := range coordinate.Roles {
				receiver.Complements[coordIndex].Roles[meaningAxis.Name] = meaningAxis.Administer(description[meaningAxis.Name], true)
			}
		}
	} else {
		for _, meaningAxis := range receiver.Roles {
			receiver.Roles[meaningAxis.Name] = meaningAxis.Administer(description[meaningAxis.Name], true)
		}
	}
}

func (receiver Meaning) Administer(administrator Meaning, isSum bool) Meaning {
	axisList := administrator.GetAxises()
	for _, axisName := range axisList {
		applied := administrator.Assets[axisName]
		if Contains(receiver.GetAxises(), axisName) && isSum { //Add, for Nouns and Sometimes Verbs

			axis := receiver.Assets[axisName]
			sum := axis.Value + applied.Value
			receiver.Assets[axisName] = Axis{Value: sum, Relative: axis.Relative, Name: axis.Name}

		} else if Contains(receiver.GetAxises(), axisName) && !isSum { //Multiply, for Modifier Words, Adjectives, Adverbs, Auxilary/Modal Verbs, sometimes actual Verbs

			axis := receiver.Assets[axisName]
			product := axis.Value * applied.Value
			receiver.Assets[axisName] = Axis{Value: product, Relative: axis.Relative, Name: axis.Name}

		} else { //Set if the axis trying to be modified by the administrator does not exist in the receiver

			axis := receiver.Assets[axisName]
			receiver.Assets[axisName] = Axis{Value: axis.Value, Relative: 0, Name: axisName}

		}
	}
	return receiver
}
