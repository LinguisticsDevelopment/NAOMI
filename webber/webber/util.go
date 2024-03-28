package main

import (
	"path/filepath"

	"github.com/smileart/lemmingo"
)

var Lem *lemmingo.Lemmingo

func Init() {
	// path, _ := filepath.Abs("/home/will/Downloads/dict/en.lmm")
	path, _ := filepath.Abs("en.lmm")
	Lem, _ = lemmingo.New(path, "", "", false, false, false)

	//wn, _ = wordnet.Parse("/home/will/Downloads/dict")
}

// https://universaldependencies.org/tagset-conversion/en-penn-uposf.html
func PennToUniv(penn string) Tag {
	switch penn {
	case "#":
		return POS_SYM
	case "$":
		return POS_SYM
	case "''":
		return POS_PUNCT
	case ",":
		return POS_PUNCT
	case "-LRB-":
		return POS_PUNCT
	case "-RRB-":
		return POS_PUNCT
	case ".":
		return POS_PUNCT
	case ":":
		return POS_PUNCT
	case "AFX":
		return POS_ADJ
	case "CC":
		return POS_CCONJ
	case "CD":
		return POS_NUM
	case "DT":
		return POS_DET
	case "EX":
		return POS_PRON
	case "FW":
		return POS_X
	case "HYPH":
		return POS_PUNCT
	case "IN":
		return POS_ADP
	case "JJ":
		return POS_ADJ
	case "JJR":
		return POS_ADJ
	case "JJS":
		return POS_ADJ
	case "LS":
		return POS_X
	case "MD":
		return POS_VERB
	case "NIL":
		return POS_X
	case "NN":
		return POS_NOUN
	case "NNP":
		return POS_PROPN
	case "NNPS":
		return POS_PROPN
	case "NNS":
		return POS_NOUN
	case "PDT":
		return POS_DET
	case "POS":
		return POS_PART
	case "PRP":
		return POS_PRON
	case "PRP$":
		return POS_DET
	case "RB":
		return POS_ADV
	case "RBR":
		return POS_ADV
	case "RBS":
		return POS_ADV
	case "RP":
		return POS_ADP
	case "SYM":
		return POS_SYM
	case "TO":
		return POS_PART
	case "UH":
		return POS_INTJ
	case "VB":
		return POS_VERB
	case "VBD":
		return POS_VERB
	case "VBG":
		return POS_VERB
	case "VBN":
		return POS_VERB
	case "VBP":
		return POS_VERB
	case "VBZ":
		return POS_VERB
	case "WDT":
		return POS_DET
	case "WP":
		return POS_PRON
	case "WP$":
		return POS_DET
	case "WRB":
		return POS_ADV
	case "``":
		return POS_PUNCT
	}

	panic("This is not a valid penn tag.")
}
