package words

import (
	"math"
)

type Component struct {
	Applied int
	Name    string
	Angle   float64
}
type Vector struct {
	Magnitude float64
	Direction map[string]Component
}

func (word *Word) GetVector(fundementalName string) Vector {
	//need to have a consistent sorting mechanism
	finalVect := Vector{}
	first := true
	count := 0
	for _, axis := range word.Roles[fundementalName].Assets {
		if first {
			newComp := Component{Name: axis.Name, Angle: 0, Applied: count}
			finalVect.Magnitude = axis.Value
			finalVect.Direction[axis.Name] = newComp
			first = false
			count++
		} else {
			finalVect.Magnitude = math.Sqrt(math.Pow(finalVect.Magnitude, 2) + math.Pow(axis.Value, 2)) //length of the vector
			newComp := Component{Name: axis.Name, Applied: count}
			newComp.Angle = dotProd(finalVect, axis.Value)
			finalVect.Direction[axis.Name] = newComp
			count++
		}
	}
	return finalVect
}

func dotProd(part1 Vector, part2 float64) float64 { //returns the angle between current and added
	dot := math.Pow(part2, 2)
	sum1 := 0.0
	sum2 := part2
	for _, component := range part1.Direction {
		sum1 += math.Pow(part1.getValue(component.Name), 2)
	}
	mag1 := math.Sqrt(sum1)
	mag2 := math.Sqrt(sum2)
	mags := mag1 * mag2
	dotProd := dot / mags
	return math.Acos(dotProd)
}

func (current Vector) getAxises() []string {
	axises := []string{}
	for name := range current.Direction {
		axises = append(axises, name)
	}
	return axises
}

func (current Vector) getValue(axis string) float64 {
	thisApplied := current.Direction[axis].Applied
	thisAngle := current.Direction[axis].Angle
	sinSet := []float64{}
	for _, component := range current.Direction {
		if component.Applied > thisApplied {
			sinSet = append(sinSet, component.Angle)
		}
	}
	sinTotal := 1.0
	for _, angleVal := range sinSet {
		sinTotal = sinTotal * math.Sin(angleVal)
	}

	return current.Magnitude * math.Cos(thisAngle) * sinTotal
}

func CombineVectors(v1 Vector, v2 Vector) Vector {
	axises1 := v1.getAxises()
	axises2 := v2.getAxises()
	axisMap := map[string]float64{}
	relevantAxises := []string{}
	relevantAxises = append(relevantAxises, axises1...)
	relevantAxises = append(relevantAxises, axises2...)
	for _, axisName := range relevantAxises {
		axisMap[axisName] = v1.getValue(axisName) + v2.getValue(axisName)
	}
	finalVect := Vector{}
	first := true
	count := 0
	for axisName, axisVal := range axisMap {
		if first {
			newComp := Component{Name: axisName, Angle: 0, Applied: count}
			finalVect.Magnitude = axisVal
			finalVect.Direction[axisName] = newComp
			first = false
			count++
		} else {
			finalVect.Magnitude = math.Sqrt(math.Pow(finalVect.Magnitude, 2) + math.Pow(axisVal, 2)) //length of the vector
			newComp := Component{Name: axisName, Applied: count}
			newComp.Angle = dotProd(finalVect, axisVal)
			finalVect.Direction[axisName] = newComp
			count++
		}
	}
	return finalVect
}

func FindIntersection(v1 Meaning, v2 Meaning, p1 Meaning, p2 Meaning) Vector {
	points := Meaning{}
	p1Axises := p1.GetAxises()
	p2Axises := p2.GetAxises()
	pointAxises := append(p1Axises, p2Axises...)
	for _, axisName := range pointAxises {
		newVal := p2.Assets[axisName].Value - p1.Assets[axisName].Value
		points.Assets[axisName] = Axis{Name: axisName, Value: newVal}
	}
	vectors := Meaning{}
	v1Axises := v1.GetAxises()
	v2Axises := v2.GetAxises()
	vectorAxises := append(v1Axises, v2Axises...)
	for _, axisName := range vectorAxises {
		newVal := v1.Assets[axisName].Value - v2.Assets[axisName].Value
		vectors.Assets[axisName] = Axis{Name: axisName, Value: newVal}
	}
	xVals := Meaning{}
	totalAxises := append(pointAxises, vectorAxises...)
	for _, axisName := range totalAxises {
		newVal := points.Assets[axisName].Value / vectors.Assets[axisName].Value
		xVals.Assets[axisName] = Axis{Name: axisName, Value: newVal}
	}
	yVals := Meaning{}
	for _, axisName := range totalAxises {
		newVal := (vectors.Assets[axisName].Value * xVals.Assets[axisName].Value) + points.Assets[axisName].Value
		yVals.Assets[axisName] = Axis{Name: axisName, Value: newVal}
	}

	return Vector{} // TODO: make work lol
}
