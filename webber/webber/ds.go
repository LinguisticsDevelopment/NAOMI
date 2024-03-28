package main

import (
	"errors"

	"github.com/google/uuid"
	"github.com/jdkato/prose/v2"

	"bytes"
	"fmt"
	"log"

	"github.com/goccy/go-graphviz"
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

func TagToString(tag Tag) string {
	switch tag {
	case POS_ADJ:
		return "ADJ"
	case POS_ADP:
		return "ADP"
	case POS_ADV:
		return "ADV"
	case POS_AUX:
		return "AUX"
	case POS_CCONJ:
		return "CCONJ"
	case POS_DET:
		return "DET"
	case POS_INTJ:
		return "INTJ"
	case POS_NOUN:
		return "NOUN"
	case POS_NUM:
		return "NUM"
	case POS_PART:
		return "PART"
	case POS_PRON:
		return "PRON"
	case POS_PROPN:
		return "PROPN"
	case POS_PUNCT:
		return "PUNCT"
	case POS_SCONJ:
		return "SCONJ"
	case POS_SYM:
		return "SYM"
	case POS_VERB:
		return "VERB"
	case POS_X:
		return "X"
	}

	panic("Not a valid tag")
}

type Word struct {
	Text string
	POS  Tag
	//Eventually Light Meaning Vector
}

const ( //Nodes
	N_NOMINAL = iota
	N_VERBAL  = iota

	N_SPECIFIER  = iota
	N_DESCRIPTOR = iota
	N_MODIFIER   = iota

	N_PREP = iota
)

type NodeType int32

func NodeTypeToString(nodeType NodeType) string {
	switch nodeType {
	case N_NOMINAL:
		return "NOMINAL"
	case N_VERBAL:
		return "VERBAL"
	case N_SPECIFIER:
		return "SPECIFIER"
	case N_DESCRIPTOR:
		return "DESCRIPTOR"
	case N_MODIFIER:
		return "MODIFIER"
	case N_PREP:
		return "PREP"
	}

	panic("Not a node type")
}

type Node struct {
	Type  NodeType
	Value Word
	POS   Tag

	Connections []*Connection
}

func NewNode(tp NodeType, text string, POS Tag) *Node {
	return &Node{tp, Word{text, POS}, POS, []*Connection{}}
}

func (n *Node) AddConnection(connection *Connection) {
	n.Connections = append(n.Connections, connection)
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

	C_SUBOORDINATION_FROM = iota
	C_SUBOORDINATION_TO   = iota
	C_PREPOSITION_FROM    = iota
	C_PREPOSITION_TO      = iota
)

type ConnectionType int32

func ConnectionTypeToString(connectionType ConnectionType) string {
	switch connectionType {
	case C_SUBJECT:
		return "SUBJECT"
	case C_PREDICATE:
		return "PREDICATE"
	case C_OBJECT:
		return "OBJECT"
	case C_INDIRECT_OBJECT:
		return "INDIRECT_OBJECT"
	case C_SUBJECT_COMPLEMENT:
		return "SUBJECT_COMPLEMENT"
	case C_OBJECT_COMPLEMENT:
		return "OBJECT_COMPLEMENT"
	case C_SPECIFICATION:
		return "SPECIFICATION"
	case C_DESCRIPTION:
		return "DESCRIPTION"
	case C_MODIFICATION:
		return "MODIFICATION"
	case C_COORDINATION:
		return "COORDINATION"
	case C_SUBOORDINATION_FROM:
		return "SUBOORDINATION_FROM"
	case C_SUBOORDINATION_TO:
		return "SUBOORDINATION_TO"
	case C_PREPOSITION_FROM:
		return "PREPOSITION_FROM"
	case C_PREPOSITION_TO:
		return "PREPOSITION_TO"
	}

	panic("Not a valid connection type")
}

type Connection struct {
	Type ConnectionType

	A *Node
	B *Node
	//Nodes []*Node
}

func NewConnection(tp ConnectionType, a *Node, b *Node) *Connection {
	return &Connection{tp, a, b}
}

func Connect(tp ConnectionType, a *Node, b *Node) {
	c1 := NewConnection(tp, a, b)
	a.AddConnection(c1)
	b.AddConnection(c1)
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
	//TODO: This is where we plug my parser
	return nil, errors.New("not implemented")
}

func PrintGraph(web Graph) {
	g := graphviz.New()
	graph, _ := g.Graph()

	defer func() {
		if err := graph.Close(); err != nil {
			log.Fatal(err)
		}
		g.Close()
	}()

	root := web.Root

	var AddNode func(root *Node, parent *Node)
	AddNode = func(root *Node, parent *Node) {
		n, _ := graph.CreateNode(root.Value.Text)
		for _, connection := range root.Connections {
			if connection.A != root && connection.A != parent {
				m, _ := graph.CreateNode(connection.A.Value.Text)
				e, _ := graph.CreateEdge(uuid.New().String(), n, m)
				e.SetLabel(ConnectionTypeToString(connection.Type))
				AddNode(connection.A, root)
			}
			if connection.B != root && connection.B != parent {
				m, _ := graph.CreateNode(connection.B.Value.Text)
				e, _ := graph.CreateEdge(uuid.New().String(), n, m)
				e.SetLabel(ConnectionTypeToString(connection.Type))
				AddNode(connection.B, root)
			}
		}
	}
	AddNode(root, nil)

	// n, _ := graph.CreateNode("n")
	// m, _ := graph.CreateNode("m")
	// e, _ := graph.CreateEdge("e", n, m)
	// e.SetLabel("e")

	var buf bytes.Buffer
	if err := g.Render(graph, "dot", &buf); err != nil {
		log.Fatal(err)
	}
	fmt.Println(buf.String())
}
