//If i get it to maximum high level understanding level - make a unique encryption key for every time a new "mind"
//is created, and then destroy the key - allows for computer to have a memory without having it's mind read, and
//for privacy between user and computer to be maintained

package main

import (
	"fmt"
	"naomi/parser/cores"
	. "naomi/parser/cores/words"
	"path/filepath"
	"strings"

	//"github.com/fluhus/gostuff/nlp/wordnet"
	"github.com/jdkato/prose/v2"

	//"github.com/mattn/go-sqlite3"
	"github.com/smileart/lemmingo"
)

var lem *lemmingo.Lemmingo

//var wn *wordnet.WordNet

func Init() {
	// path, _ := filepath.Abs("/home/will/Downloads/dict/en.lmm")
	path, _ := filepath.Abs("en.lmm")
	lem, _ = lemmingo.New(path, "", "", false, false, false)

	//wn, _ = wordnet.Parse("/home/will/Downloads/dict")
}

//TODO: Tokenize Sentences to make words automatically, as well as assign standard meaning values to nouns/verbs (tense, number, ect)
//TODO: Turn a tree back into a sentence with fully tokenized words (extremely simply, i.e. if we have run(plural), I WANT it to output runed instead of ran)
//TODO: This would mostly complete Phase 1 honestly. Then onto refactoring this shit heavily, and then the hard part of making the lexicon database and vector system

type Token struct {
	word string
	part string
}

func ComplexTagToSimple(tag string) string {
	switch res := strings.ToLower(tag)[0]; res {
	case 'n':
		return "noun"
	case 'v':
		return "verb"
	case 'r':
		return "adv"
	case 'a':
		return "adj"
	case 'c':
		return "coord"
	case 'i':
		return "suboord"
	case 'p':
		return "prep"
	}

	return "unknown"
}

func main() {
	Init()

	fmt.Println("Enter Sentence (no punctuation plz): ")
	var sentenceString string

	// Taking input from user
	fmt.Scanln(&sentenceString)

	//sentenceList := strings.Split(sentenceString, " ")
	doc, _ := prose.NewDocument(sentenceString)

	var sentence []Word

	for _, tok := range doc.Tokens() {
		l, _, _ := lem.Lemma(tok.Text, tok.Tag)
		//Token{l, ComplexTagToSimple(tok.Tag)
		sentence = append(sentence, BuildWord(l, ComplexTagToSimple(tok.Tag)))
	}

	//bob := BuildWord("bob", "noun")
	//jim := BuildWord("jim", "noun")

	//kills := BuildWord("kills", "verb")
	///green := BuildWord("green", "adj")
	//extremely := BuildWord("extremely", "adv")
	//under := BuildWord("under", "prep")
	//and := BuildWord("and", "coord")
	//who := BuildWord("who", "suboord")

	//sentence := []Word{green, bob, kills, jim}
	//sentence := []Word{bob, kills, jim, under, the, sea}

	cores.TotalParse(sentence)
}
