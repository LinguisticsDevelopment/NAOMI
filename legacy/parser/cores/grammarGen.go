package cores

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
)

type Part struct {
	Distance        int    //Absolute Value away from the root
	ApplyInChain    bool   // "/"
	FindAllinDir    bool   // "*"
	SkipConsumption bool   // "_"
	TypeKind        string // Part BEFORE the "."
	SubType         string // Part AFTER the "."
}

type Rule struct {
	IsRecursive   bool // "N" or "R"
	NeedsMatching bool // "%"
	Resultant     string
	RootType      string
	RootSubType   string
	Before        []Part
	After         []Part
}

func parseRule(input string, result string) Rule {
	isRec := false
	if strings.HasPrefix(input, "R") {
		isRec = true
	}
	input = input[2:]
	needsMatching := false
	if strings.HasPrefix(input, "%") {
		needsMatching = true
		input = input[2:]
	}

	rights := strings.SplitAfter(input, ">")
	lefts := []string{}
	for _, str := range rights {
		newSplit := strings.SplitAfter(str, "<")
		lefts = append(lefts, newSplit...)
	}
	totals := []string{}
	for _, str := range lefts {
		newSplit := strings.SplitAfter(str, "?")
		totals = append(totals, newSplit...)
	}
	befores := []string{}
	root := ""
	afters := []string{}
	for _, rule := range totals {
		if strings.HasSuffix(rule, ">") {
			befores = append(befores, rule[:len(rule)-2])
		} else if strings.HasSuffix(rule, "<") {
			afters = append(afters, rule[:len(rule)-2])
		} else if strings.HasSuffix(rule, "?") {
			root = rule[:len(rule)-2]
		}
	}
	beforeParts := []Part{}
	afterParts := []Part{}
	for index, before := range befores {
		newPart := Part{Distance: len(befores) - index, FindAllinDir: false, ApplyInChain: false, SkipConsumption: false}
		clean := strings.Fields(before)
		for _, aspect := range clean {
			if aspect == "*" {
				newPart.FindAllinDir = true
			} else if aspect == "/" {
				newPart.ApplyInChain = true
			} else if aspect == "_" {
				newPart.SkipConsumption = true
			} else {
				div := strings.Split(aspect, ".")
				newPart.TypeKind = div[0]
				if len(div) > 1 {
					newPart.SubType = div[1]
				} else {
					newPart.SubType = ""
				}
			}
		}
		beforeParts = append(beforeParts, newPart)
	}
	for index, after := range afters {
		newPart := Part{Distance: index + 1, FindAllinDir: false, ApplyInChain: false, SkipConsumption: false}
		clean := strings.Fields(after)
		for _, aspect := range clean {
			if aspect == "*" {
				newPart.FindAllinDir = true
			} else if aspect == "/" {
				newPart.ApplyInChain = true
			} else if aspect == "_" {
				newPart.SkipConsumption = true
			} else {
				div := strings.Split(aspect, ".")
				newPart.TypeKind = div[0]
				if len(div) > 1 {
					newPart.SubType = div[1]
				} else {
					newPart.SubType = ""
				}
			}
		}
		afterParts = append(afterParts, newPart)
	}
	root = strings.TrimSpace(root)
	newRule := Rule{IsRecursive: isRec, Before: beforeParts, After: afterParts, Resultant: result, NeedsMatching: needsMatching}
	div := strings.Split(root, ".")
	newRule.RootType = div[0]
	if len(div) > 1 {
		newRule.RootSubType = div[1]
	} else {
		newRule.RootSubType = ""
	}
	return newRule
}

func ruleSetParse(file string) [][]Rule {
	jsonFile, err := os.Open(file)
	if err != nil {
		fmt.Println(err)
	}

	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	var ruleset map[string]map[int]string

	json.Unmarshal(byteValue, &ruleset)

	trueRules := [][]Rule{}
	order := strings.Fields(ruleset["order"][0])

	for _, partOfSpeech := range order {
		PartRules := ruleset[partOfSpeech]
		thisPartsRules := []Rule{}
		for workingRule := 1; workingRule < len(PartRules); workingRule++ {
			nextRule := parseRule(PartRules[workingRule], PartRules[0])
			thisPartsRules = append(thisPartsRules, nextRule)
		}
		trueRules = append(trueRules, thisPartsRules)
	}
	return trueRules
}
