package words

type Axis struct {
	Name     string
	Value    float64
	Relative float64
}

type Meaning struct {
	Name     string
	Relevant bool
	Affected *Word           //Primarily for Verbs, what they are storing
	Assets   map[string]Axis //Nouns and Adjectives - Meaning
}

func BuildMeaning(defName string, defRelevance bool, axisNames []string, defaultValue float64) Meaning {
	newPart := Meaning{}
	newAssets := BuildAxises(axisNames, defaultValue)

	newPart.Name = defName
	newPart.Relevant = defRelevance
	newPart.Affected = &Word{}
	newPart.Assets = newAssets

	return newPart
}

func BuildAxises(labels []string, defaultVal float64) map[string]Axis {
	newMap := map[string]Axis{}
	for _, word := range labels {
		newMap[word] = Axis{Value: defaultVal, Relative: 0, Name: word}
	}
	return newMap
}

func (current *Meaning) GetAxises() []string {
	pool := current.Assets
	axises := make([]string, len(pool))
	i := 0
	for axis := range pool {
		axises[i] = axis
		i++
	}
	return axises
}

func Contains(s []string, str string) bool { //for axis checks in word / part of speech methods
	for _, v := range s {
		if v == str {
			return true //returns true (does contain) if it finds a matching string
		}
	}
	return false //returns false (doesn't contain) if all fails
}
