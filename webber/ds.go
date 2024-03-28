package main

import (
	"errors"

	"github.com/jdkato/prose/v2"
)

// https://universaldependencies.org/u/pos/
//Descriptors - adjectives, numerals, determiners - Affect Nominals
//Specifiers - adverbs, negator, affirmer - Affect Scopes
//Modifiers - Modals & Auxilaries(Maybe) - Affect Roles or Scopes

//Nominal: Nouns, Pronouns
//Verbal: Verbs acting as Actions
//Specifiers:
//Modifiers
//Descriptors: Adjectives, Determiners(the, an, one)

const ( //Parts of Speech
	POS_ADJ   = iota
	POS_ADP   = iota
	POS_ADV   = iota
	POS_AUX   = iota
	POS_CCONJ = iota
	POS_DET   = iota
	POS_INTJ  = iota
	POS_NOUN  = iota
	POS_NUM   = iota
	POS_PART  = iota
	POS_PRON  = iota
	POS_PROPN = iota
	POS_PUNCT = iota
	POS_SCONJ = iota
	POS_SYM   = iota
	POS_VERB  = iota
	POS_X     = iota
)

type Tag int32

type Word struct {
	Text string
	POS  Tag
	//Eventually Light Meaning Vector
}

const ( //Nodes
	N_NOMINAL = iota
	N_VERB    = iota

	N_SPECIFIER  = iota
	N_DESCRIPTOR = iota
	N_MODIFIER   = iota
)

type NodeType int32

type Node struct {
	Type  NodeType
	Value Word
	POS   Tag

	Connections []*Connection
}

//Connections:
//	-Connectors
//  -Type
//  -Applier(?)
//
// Subject, Object, Indirect Object - ARE CONNECTED TO THE ROOT VERB

//Words

//There is 2 datatypes
//Nodes - Nouns, Adjectives, Verbs, Adverbs, and Coordinating Conjunctions
//Edges - Prepositions, Suboordinating Clauses, Logical Operators, and Parts of a Sentence

const (
	//Sentence Parts
	C_SUBJECT            = iota
	C_PREDICATE          = iota
	C_OBJECT             = iota
	C_INDIRECT_OBJECT    = iota
	C_SUBJECT_COMPLEMENT = iota
	C_OBJECT_COMPLEMENT  = iota

	C_SPECIFICATION = iota
	C_DESCRIPTION   = iota
	C_MODIFICATION  = iota

	C_COORDINATION = iota

	C_SUBOORDINATION = iota
	C_PREPOSITION    = iota
)

type ConnectionType int32

type Connection struct {
	Type  ConnectionType
	Value Word

	Nodes []*Node
}

type Graph struct {
	Sentence []*Word
	Root     *Node
}

func NewGraph(sentence string) (Graph, error) {
	doc, _ := prose.NewDocument(sentence)

	var words []*Word

	for _, tok := range doc.Tokens() {
		l, _, _ := Lem.Lemma(tok.Text, tok.Tag)

		words = append(words, &Word{l, PennToUniv(tok.Tag)})
	}

	root, err := Parse(words)
	if err != nil {
		return Graph{words, nil}, err
	}

	return Graph{words, root}, nil
}

func Parse(sentence []*Word) (*Node, error) {
	return nil, errors.New("not implemented")
}
