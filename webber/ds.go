package main

import (
	"os"
	"unicode"

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

var TagToString = map[Tag]string{
	POS_ADJ:   "ADJ",
	POS_ADP:   "ADP",
	POS_ADV:   "ADV",
	POS_AUX:   "AUX",
	POS_CCONJ: "CCONJ",
	POS_DET:   "DET",
	POS_INTJ:  "INT",
	POS_NOUN:  "NOUN",
	POS_NUM:   "NUM",
	POS_PART:  "PART",
	POS_PRON:  "PRON",
	POS_PROPN: "PROPN",
	POS_PUNCT: "PUNCT",
	POS_SCONJ: "SCONJ",
	POS_SYM:   "SYM",
	POS_VERB:  "VERB",
	POS_X:     "X",
}

const ( //Nodes
	N_NIL = -1

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

	N_INTJ = iota
)

type NodeType int32

var TagToNodeType = map[Tag]NodeType{
	POS_ADJ:   N_DESCRIPTOR,
	POS_ADP:   N_PREP,
	POS_ADV:   N_SPECIFIER,
	POS_AUX:   N_MODIFIER,
	POS_CCONJ: N_COORD,
	POS_DET:   N_DESCRIPTOR,
	POS_INTJ:  N_INTJ,
	POS_NOUN:  N_NOUN,
	POS_NUM:   N_DESCRIPTOR,
	POS_PART:  N_MODIFIER,
	POS_PRON:  N_NOUN,
	POS_PROPN: N_NOUN,
	POS_PUNCT: N_NIL,
	POS_SCONJ: N_SUBOORD,
	POS_SYM:   N_NIL,
	POS_VERB:  N_VERBAL,
	POS_X:     N_NIL,
}

var StringToNodeType = map[string]NodeType{
	"NIL": N_NIL,

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

	"INTJ": N_INTJ,
}

var NodeTypeToString = map[NodeType]string{
	N_NIL: "NIL",

	N_NOUN:       "NOUN",
	N_VERBAL:     "VERBAL",
	N_NOMINAL:    "NOMINAL",
	N_PREDICATE:  "PREDICATE",
	N_CLAUSE:     "CLAUSE",
	N_SPECIFIER:  "SPECIFIER",
	N_DESCRIPTOR: "DESCRIPTOR",
	N_MODIFIER:   "MODIFIER",
	N_COORD:      "COORD",
	N_SUBOORD:    "SUBOORD",

	N_PREP:      "PREP",
	N_PREP_SPEC: "PREP_SPEC",
	N_PREP_DESC: "PREP_DESC",
	N_PREP_MIX:  "PREP_MIX",
	N_PP_NOUN:   "PP_NOUN",
	N_PP_MIX:    "PP_MIX",
	N_PP_DESC:   "PP_DESC",
	N_PP_SPEC:   "PP_SPEC",
	N_PP_VERB:   "PP_VERB",

	N_INTJ: "INTJ",
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

var SubTypeToString = map[SubType]string{
	S_SUBOORDINATE: "SUBOORDINATE",
	S_INDEPENDENT:  "INDEPENDENT",
	S_NOMINAL:      "NOMINAL",
	S_MODAL:        "MODAL",
	S_COMPARATIVE:  "COMPARATIVE",
	S_SUPERLATIVE:  "SUPERLATIVE",
	S_P_MIX:        "P_MIX",
	S_P_DESC:       "P_DESC",
	S_P_SPEC:       "P_SPEC",
	S_P_VERB:       "P_VERB",
	S_P_NORM:       "P_NORM",
	S_MASCULINE:    "MASCULINE",
	S_FEMININE:     "FEMININE",
	S_SINGULAR:     "SINGULAR",
	S_PLURAL:       "PLURAL",
}

type Word struct {
	Text string
	POS  Tag
	//Eventually Light Meaning Vector
}

type Node struct {
	Type  NodeType
	Value *Word
	POS   Tag

	Connections []*Connection
}

func NewNode(tp NodeType, text string, POS Tag) *Node {
	return &Node{tp, &Word{text, POS}, POS, []*Connection{}}
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

var StringToConnectionType = map[string]ConnectionType{
	"SUBJECT":             C_SUBJECT,
	"PREDICATE":           C_PREDICATE,
	"OBJECT":              C_OBJECT,
	"INDIRECT_OBJECT":     C_INDIRECT_OBJECT,
	"SUBJECT_COMPLEMENT":  C_SUBJECT_COMPLEMENT,
	"OBJECT_COMPLEMENT":   C_OBJECT_COMPLEMENT,
	"SPECIFICATION":       C_SPECIFICATION,
	"DESCRIPTION":         C_DESCRIPTION,
	"MODIFICATION":        C_MODIFICATION,
	"COORDINATION":        C_COORDINATION,
	"SUBOORDINATION_FROM": C_SUBOORDINATION_FROM,
	"SUBOORDINATION_TO":   C_SUBOORDINATION_TO,
	"PREPOSITION_FROM":    C_PREPOSITION_FROM,
	"PREPOSITION_TO":      C_PREPOSITION_TO,
}

var ConnectionTypeToString = map[ConnectionType]string{
	C_SUBJECT:             "SUBJECT",
	C_PREDICATE:           "PREDICATE",
	C_OBJECT:              "OBJECT",
	C_INDIRECT_OBJECT:     "INDIRECT_OBJECT",
	C_SUBJECT_COMPLEMENT:  "SUBJECT_COMPLEMENT",
	C_OBJECT_COMPLEMENT:   "OBJECT_COMPLEMENT",
	C_SPECIFICATION:       "SPECIFICATION",
	C_DESCRIPTION:         "DESCRIPTION",
	C_MODIFICATION:        "MODIFICATION",
	C_COORDINATION:        "COORDINATION",
	C_SUBOORDINATION_FROM: "SUBOORDINATION_FROM",
	C_SUBOORDINATION_TO:   "SUBOORDINATION_TO",
	C_PREPOSITION_FROM:    "PREPOSITION_FROM",
	C_PREPOSITION_TO:      "PREPOSITION_TO",
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

func Connect(tp ConnectionType, a *Node, b *Node) *Connection {
	c1 := NewConnection(tp, a, b)
	a.AddConnection(c1)
	b.AddConnection(c1)

	return c1
}

type Web struct {
	Sentence []*Word
	Root     *Node
}

func NewWeb(sentence string) (Web, error) {
	doc, _ := prose.NewDocument(sentence)

	var words []*Word

	for i, tok := range doc.Tokens() {
		// l, _, _ := Lem.Lemma(tok.Text, tok.Tag)
		l := tok.Text
		l += fmt.Sprintf(" (%d)", i)
		fmt.Fprintf(os.Stderr, "%s %s\n", tok.Text, tok.Tag)

		if !unicode.IsPunct([]rune(tok.Tag)[0]) {
			words = append(words, &Word{l, PennToUniv(tok.Tag)})
		}
	}

	root, err := Parse(words)
	if err != nil {
		return Web{words, nil}, err
	}

	return Web{words, root}, nil
}

func Parse(sentence []*Word) (*Node, error) {
	ParserInit()

	nodes := make([]*Node, 0, len(sentence))
	for _, word := range sentence {
		nodes = append(nodes, &Node{TagToNodeType[word.POS], word, word.POS, []*Connection{}})
	}

	root, err := SentenceParse(nodes)
	return root, err
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
		// n, _ := graph.CreateNode(uuid.New().String())
		// n.SetLabel(root.Value.Text)
		for _, connection := range root.Connections {
			if connection.A != root && connection.A != parent {
				m, _ := graph.CreateNode(connection.A.Value.Text)
				// m, _ := graph.CreateNode(uuid.New().String())
				// m.SetLabel(connection.A.Value.Text)
				e, _ := graph.CreateEdge(uuid.New().String(), n, m)
				e.SetLabel(ConnectionTypeToString[connection.Type])
				AddNode(connection.A, root)
			}
			if connection.B != root && connection.B != parent {
				m, _ := graph.CreateNode(connection.B.Value.Text)
				// m, _ := graph.CreateNode(uuid.New().String())
				// m.SetLabel(connection.B.Value.Text)
				e, _ := graph.CreateEdge(uuid.New().String(), n, m)
				e.SetLabel(ConnectionTypeToString[connection.Type])
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
