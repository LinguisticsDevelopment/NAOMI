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

var StringToTag = map[string]Tag{
	"ADJ":   POS_ADJ,
	"ADP":   POS_ADP,
	"ADV":   POS_ADV,
	"AUX":   POS_AUX,
	"CCONJ": POS_CCONJ,
	"DET":   POS_DET,
	"INT":   POS_INTJ,
	"NOUN":  POS_NOUN,
	"NUM":   POS_NUM,
	"PART":  POS_PART,
	"PRON":  POS_PRON,
	"PROPN": POS_PROPN,
	"PUNCT": POS_PUNCT,
	"SCONJ": POS_SCONJ,
	"SYM":   POS_SYM,
	"VERB":  POS_VERB,
	"X":     POS_X,
}

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

const ( //Nodes
	N_NOUN   = iota
	N_VERBAL = iota

	N_PREDICATE = iota
	N_NOMINAL   = iota
	N_CLAUSE    = iota

	N_SPECIFIER  = iota
	N_DESCRIPTOR = iota
	N_MODIFIER   = iota

	N_COORD   = iota
	N_SUBOORD = iota

	N_PREP      = iota
	N_PREP_SPEC = iota //this should eventually use the subtype format in the grammar file and in the generator and parser
	N_PREP_DESC = iota
	N_PREP_MIX  = iota

	N_PP_NOUN = iota
	N_PP_MIX  = iota
	N_PP_DESC = iota
	N_PP_SPEC = iota
	N_PP_VERB = iota
)

type NodeType int32

var StringToNodeType = map[string]NodeType{
	"NOUN":       N_NOUN,
	"VERBAL":     N_VERBAL,
	"NOMINAL":    N_NOMINAL,
	"PREDICATE":  N_PREDICATE,
	"CLAUSE":     N_CLAUSE,
	"SPECIFIER":  N_SPECIFIER,
	"DESCRIPTOR": N_DESCRIPTOR,
	"MODIFIER":   N_MODIFIER,
	"COORD":      N_COORD,
	"SUBOORD":    N_SUBOORD,

	"PREP":      N_PREP,
	"PREP_SPEC": N_PREP_SPEC,
	"PREP_DESC": N_PREP_DESC,
	"PREP_MIX":  N_PREP_MIX,
	"PP_NOUN":   N_PP_NOUN,
	"PP_MIX":    N_PP_MIX,
	"PP_DESC":   N_PP_DESC,
	"PP_SPEC":   N_PP_SPEC,
	"PP_VERB":   N_PP_VERB,
}

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

const (
	//clause types
	S_SUBOORDINATE = iota
	S_INDEPENDENT  = iota

	//verb forms
	S_NOMINAL = iota
	S_MODAL   = iota

	//Descriptor/Specifier forms
	S_COMPARATIVE = iota
	S_SUPERLATIVE = iota

	//Preposition/Prepositional Phrase forms TODO: IMPLEMENT THESE PLEASE FOR THE LOVE OF GOD
	S_P_MIX  = iota
	S_P_DESC = iota
	S_P_SPEC = iota
	S_P_VERB = iota
	S_P_NORM = iota

	//gender
	S_MASCULINE = iota
	S_FEMININE  = iota

	//number
	S_SINGULAR = iota
	S_PLURAL   = iota
)

type SubType int32

//This is being partially implemented to help with clauses and with verbs
//But can eventually be used to implement other languages (think romantic languages)
//To allow for gender/number/case implementation
//Still need to modify the parser and grammar generator
//to allow for this to be compared correctly
//but this is a start
//PROBABLY have each language grammar file initially denote what subTypes are present
//Point is this eventually allows it to not be only reliant on order for other good langauges lol

var StringToSubType = map[string]SubType{
	"SUBOORDINATE": S_SUBOORDINATE,
	"INDEPENDENT":  S_INDEPENDENT,
	"NOMINAL":      S_NOMINAL,
	"MODAL":        S_MODAL,
	"COMPARATIVE":  S_COMPARATIVE,
	"SUPERLATIVE":  S_SUPERLATIVE,
	"P_MIX":        S_P_MIX,
	"P_DESC":       S_P_DESC,
	"P_SPEC":       S_P_SPEC,
	"P_VERB":       S_P_VERB,
	"P_NORM":       S_P_NORM,
	"MASCULINE":    S_MASCULINE,
	"FEMININE":     S_FEMININE,
	"SINGULAR":     S_SINGULAR,
	"PLURAL":       S_PLURAL,
}

type Word struct {
	Text string
	POS  Tag
	//Eventually Light Meaning Vector
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

type Web struct {
	Sentence []*Word
	Root     *Node
}

func NewGraph(sentence string) (Web, error) {
	doc, _ := prose.NewDocument(sentence)

	var words []*Word

	for _, tok := range doc.Tokens() {
		l, _, _ := Lem.Lemma(tok.Text, tok.Tag)

		words = append(words, &Word{l, PennToUniv(tok.Tag)})
	}

	root, err := Parse(words)
	if err != nil {
		return Web{words, nil}, err
	}

	return Web{words, root}, nil
}

func Parse(sentence []*Word) (*Node, error) {
	//TODO: This is where we plug my parser
	return nil, errors.New("not implemented")
}

func PrintGraph(web Web) {
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
