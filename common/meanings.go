package common

type Axis struct {
	Name     string  //The Key of this Axis
	Value    float64 //Actual Value
	Relative float64 //Relative Modification after relatives are calculated
}

type Meaning struct {
	Name       string          //The Key of this Meaning
	Relevant   bool            //If it is used by the idea it is in
	Dimensions map[string]Axis //The actual Axises assigned to their keys
}
