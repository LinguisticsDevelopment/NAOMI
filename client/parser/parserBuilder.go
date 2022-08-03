package parser

import (
	"fmt"
	"io/ioutil"
	"os"

	"gopkg.in/yaml.v3"
)

type Part struct {
	Direction     int  //Distance and Direction
	AllUntilGap   bool //Go until we find a
	CaseMatters   bool
	IsConsumed    bool
	Case          string
	ModType       string
	AspectApplied int
}

type Rule struct {
	CaseMatch   string // "%"
	GenMatch    string
	RootType    string
	RootSubType string
	Parts       []Part
}

type RuleSet map[string][]Rule

func build(gramFile string) RuleSet {
	yamlFile, err := os.Open(gramFile)
	if err != nil {
		fmt.Println(err)
	}
	defer yamlFile.Close()

	byteVal, _ := ioutil.ReadAll(yamlFile)

	var theseRules RuleSet

	err2 := yaml.Unmarshal(byteVal, &theseRules)
	if err2 != nil {
		fmt.Println(err2)
	}
	return theseRules
}
