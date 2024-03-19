package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/fluhus/gostuff/nlp/wordnet"
	"github.com/jdkato/prose/v2"
	_ "github.com/mattn/go-sqlite3"
	"github.com/smileart/lemmingo"
)

var lem *lemmingo.Lemmingo
var wn *wordnet.WordNet

func Init() {
	// path, _ := filepath.Abs("/home/will/Downloads/dict/en.lmm")
	path, _ := filepath.Abs("en.lmm")
	lem, _ = lemmingo.New(path, "", "", false, false, false)

	wn, _ = wordnet.Parse("/home/will/Downloads/dict")
}

func ComplexTagToSimple(tag string) string {
	res := strings.ToLower(tag)[0]

	if res == 'n' || res == 'v' || res == 'r' || res == 'a' {
		return string(res)
	}

	return "unknown"
}

type Token struct {
	word string
	pos  string
}

func (w *Word) equals(other *Word) bool {
	return w.word == other.word && w.pos == other.pos
}

func ParseDefinition(def string) []Token {
	doc, _ := prose.NewDocument(def)

	var res []Token

	for _, tok := range doc.Tokens() {
		l, _, _ := lem.Lemma(tok.Text, tok.Tag)
		res = append(res, Token{l, ComplexTagToSimple(tok.Tag)})
	}

	return res
}

func PrintDefinition(def []Token) {
	for _, tok := range def {
		fmt.Printf("%s ", tok.word)
	}
	fmt.Println()
}

type Word struct {
	word string
	pos  string
}

func (w *Word) Key() string {
	return w.word + "-" + w.pos
}

var count int = 0

var seen map[string]bool = make(map[string]bool)
var elemental map[string]int = make(map[string]int)

func Store(word Word) {
	_, ok := elemental[word.Key()]
	if ok {
		elemental[word.Key()] += 1
	} else {
		elemental[word.Key()] = 1
		// fmt.Println(word.word, word.pos)
	}
}

func Drill(word Word) {
	if word.pos == "unknown" {
		return
	}

	// if count > 100 {
	// 	return
	// }

	count += 1

	// fmt.Println()
	// for k, v := range seen {
	// 	fmt.Println(k, "value is", v)
	// }
	// fmt.Println()

	if seen[word.Key()] {
		Store(word)
		return
	}

	seen[word.Key()] = true

	// fmt.Println("DEBUG:", word.word, word.pos)

	outputs := wn.SearchRanked(word.word)[word.pos]
	if len(outputs) <= 0 {
		Store(word)
		return
	}

	// output := outputs[0]
	for _, output := range outputs {
		def := strings.Split(output.Gloss, ";")[0]

		toks := ParseDefinition(def)
		// PrintDefinition(toks)

		for _, tok := range toks {
			w := Word{tok.word, tok.pos}
			Drill(w)
		}
	}
}

func main() {
	Init()

	file, _ := os.Open("./allwords.txt")
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		parts := strings.Split(scanner.Text(), "|")
		word := parts[0]
		pos := parts[1]

		w := Word{word, pos}
		Drill(w)
	}

	// fmt.Println()

	for k, v := range elemental {
		fmt.Println(k, v)
	}
}
